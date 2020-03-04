# Ablation1 1: Without SNLI pre trained sentence encoding, instead we use doc2vec
#		    2: With Using Decomposable attention on sentence level embeddings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import xml.etree.ElementTree as ET
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import pickle
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from nltk import word_tokenize, sent_tokenize
from keras.callbacks import ModelCheckpoint
from spacy_decomposable_attention import _BiRNNEncoding
from spacy_decomposable_attention import _Attention
from spacy_decomposable_attention import _SoftAlignment
from spacy_decomposable_attention import _Comparison
from spacy_decomposable_attention import _Entailment, _StaticEmbedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Flatten,Bidirectional, GRU, LSTM, TimeDistributed, Embedding
from keras import backend as K
import time
from gensim.models.doc2vec import Doc2Vec
pv_model = Doc2Vec.load("../../enwiki_dbow/doc2vec.bin")
from nltk.corpus import stopwords
import string
stopwords = list(string.punctuation)+list(set(stopwords.words('english')))

def doc_to_mat(docs,max_sents):
	if_word_in_stopwords = defaultdict(int)
	for word in stopwords:
		if_word_in_stopwords[word] = 1
	mat = np.zeros((len(docs),max_sents,SENT_DIM),dtype="float32")
	for i in range(len(docs)):
		docs[i] = docs[i][:max_sents]
		for j in range(len(docs[i])):
			words = [word for word in docs[i][j] if if_word_in_stopwords[word]==0]
			sent_vec = pv_model.infer_vector(doc_words=words, alpha=0.1, min_alpha=0.0001, steps=5)
			mat[i,max_sents-len(docs[i])+j] = sent_vec
	return mat


def parse_file(file):
	sentences = sent_tokenize(open(file,"r").read().decode("ascii","ignore"))
	data = [word_tokenize(sent) for sent in sentences]
	return data


SENT_DIM = 300
NUM_CLASSES = 2
targets = list()
sources = list()
gold = list()
topics_allowed="q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
topics_allowed=topics_allowed.split(", ")
for line in open("redundancy_list.txt","r"):
	tokens = line.split()
	if tokens[0] in topics_allowed:
		targets.append([word_tokenize(sent) for sent in sent_tokenize(open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[1],"r").read().decode("utf-8","ignore"))])
		sources.append([word_tokenize(sent) for sent in sent_tokenize(" . ".join([open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[i],"r").read().decode("utf-8","ignore") for i in range(2,len(tokens))]))])
		# 1 for novel 0 for non-novel
		gold.append(0)
for line in open("novel_list.txt","r"):
	tokens = line.split()
	if tokens[0] in topics_allowed:
		targets.append([word_tokenize(sent) for sent in sent_tokenize(open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[1],"r").read().decode("utf-8","ignore"))])
		sources.append([word_tokenize(sent) for sent in sent_tokenize(" . ".join([open("apwsj_parsed_documents/"+tokens[0]+"/"+tokens[i],"r").read().decode("utf-8","ignore") for i in range(2,len(tokens))]))])
		# 1 for novel 0 for non-novel
		gold.append(1)
max_sents = 336
print("Max sentences in a document: "+str(max_sents))

print("Total no. of instances: "+str(len(gold)))
print("Total no. of Novel instances: "+str(sum(gold)))
print("Total no. of Non-Novel instances: "+str(len(gold)-sum(gold)))
print("Percentage of Novel instances: "+str((sum(gold)/float(len(gold)))*100)+"%")

gold_list = np.array([i for i in gold])
sources = np.array(sources)
targets = np.array(targets)

kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 8309)
save_file = "apwsj_ablation1_ten_fold_progress.p"
fold = 1
if os.path.exists(save_file):
	predictions,golds,folds_complete,target,source,attentions,probs = pickle.load(open(save_file,"rb"))
else:
	predictions = []
	golds = []
	folds_complete = 0
	target = []
	source = []
	attentions = []
	probs = []

for train,test in kfold.split(np.zeros(len(gold_list)),gold_list):
	if fold <= folds_complete:
		fold+=1
		continue
	print("\nFold: "+str(fold))
	fold+=1
	print("\nCompiling model\n")
	tgt_in = Input(shape=(max_sents,SENT_DIM), dtype='float32')
	srcs_in = Input(shape=(max_sents,SENT_DIM), dtype='float32')
	tgt_vec = tgt_in
	src_vec = srcs_in
	attention = _Attention(max_sents,SENT_DIM,dropout=0.2)(tgt_vec,src_vec)
	align1= _SoftAlignment(max_sents,SENT_DIM)(src_vec,attention)
	align2= _SoftAlignment(max_sents,SENT_DIM)(tgt_vec,attention,transpose=True)
	vec_l = _Comparison(max_sents,SENT_DIM,dropout=0.2)(tgt_vec,align1)
	vec_r = _Comparison(max_sents,SENT_DIM,dropout=0.2)(src_vec,align2)
	pds = _Entailment(SENT_DIM,NUM_CLASSES,dropout=0.2)(vec_l,vec_r)
	model = Model(inputs=[tgt_in,srcs_in],outputs=pds)
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.001),metrics=["accuracy"])

	get_attention_matrix = K.function([model.layers[0].input,model.layers[1].input,K.learning_phase()],[model.layers[3].output])

	NUM_EPOCHS = 30
	BATCH_SIZE = 25

	st = time.time()
	train_tgt_vec = doc_to_mat(targets[train],max_sents)
	train_source_vec = doc_to_mat(sources[train],max_sents)
	print("Training data created in (seconds): "+str(time.time()-st))
	train_gold = to_categorical(gold_list[train],num_classes=NUM_CLASSES)
	test_tgt_vec = doc_to_mat(targets[test],max_sents)
	test_source_vec = doc_to_mat(sources[test],max_sents)
	print("Total data created in (seconds): "+str(time.time()-st))
	test_gold = to_categorical(gold_list[test],num_classes=NUM_CLASSES)
	
	cb = [ModelCheckpoint("temp4_model.hdf5",monitor="val_loss",verbose=1,save_best_only=True,save_weights_only=True)]

	print("\nTraining model\n")
	history = model.fit(x=[train_tgt_vec,train_source_vec],y=train_gold,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,verbose=1,validation_split=0.1,shuffle=True,callbacks=cb)
	model.load_weights("temp4_model.hdf5")

	preds = model.predict(x=[test_tgt_vec,test_source_vec],batch_size=BATCH_SIZE,verbose=1)
	probs.append(preds)
	preds = np.argmax(preds,axis=1)
	gold_test = np.argmax(test_gold,axis=1)
	predictions.append(preds)
	golds.append(gold_test)
	target.append(targets[test])
	source.append(sources[test])
	att = get_attention_matrix([test_tgt_vec,test_source_vec,0])[0]
	attentions.append(att)
	p,r,f,_ = precision_recall_fscore_support(gold_test,preds,labels=[0,1])
	test_acc = accuracy_score(gold_test,preds)
	print("Testing accuracy: "+str(test_acc))
	print("Confusion matrix: \n"+str(confusion_matrix(gold_test,preds,labels=[0,1])))
	print("Precision: "+str(p))
	print("Recall: "+str(r))
	print("F-score: "+str(f))

predictions = [i for l in predictions for i in l]
golds = [i for l in golds for i in l]
classes = [0,1]
p,r,f,_ = precision_recall_fscore_support(golds,predictions,labels=classes)
test_acc = accuracy_score(golds,predictions)
print("Testing accuracy: "+str(test_acc))
print("Confusion matrix: \n"+str(confusion_matrix(golds,predictions,labels=classes)))
print("Precision: "+str(p))
print("Recall: "+str(r))
print("F-score: "+str(f))
pickle.dump([predictions,golds,fold-1,target,source,attentions,probs],open(save_file,"wb"))