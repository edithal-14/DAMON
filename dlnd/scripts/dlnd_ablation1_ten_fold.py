# Ablation1 1: Without SNLI pre trained sentence encoding, instead we use doc2vec
#		    2: With Using Decomposable attention on sentence level embeddings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
	#global vocab
	# if_word_in_vocab = defaultdict(int)
	# for word in vocab:
	# 	if_word_in_vocab[word]=1
	if_word_in_stopwords = defaultdict(int)
	for word in stopwords:
		if_word_in_stopwords[word] = 1
	mat = np.zeros((len(docs),max_sents,SENT_DIM),dtype="float32")
	for i in range(len(docs)):
		docs[i] = docs[i][:max_sents]
		for j in range(len(docs[i])):
			words = [word for word in docs[i][j] if if_word_in_stopwords[word]==0]
			#sent_vec = np.mean(np.array([vocab[word] for word in words]),axis=0)
			sent_vec = pv_model.infer_vector(doc_words=words, alpha=0.1, min_alpha=0.0001, steps=5)
			mat[i,max_sents-len(docs[i])+j] = sent_vec
	return mat


def parse_file(file):
	sentences = sent_tokenize(open(file,"r").read().decode("ascii","ignore"))
	data = [word_tokenize(sent) for sent in sentences]
	return data

# def fetch_glove_embeddings(docs,glove_path):
# 	print("Fetching glove embeddings")
# 	vocab = defaultdict(int)
# 	for doc in docs:
# 		for sent in doc:
# 			for word in sent:
# 				vocab[word] = 1
# 	vocab_size = len(vocab)
# 	word_emb = dict()
# 	with open(glove_path,"r") as f:
# 		for line in f:
# 			word,vec = line.split(' ',1)
# 			if vocab[word] == 1:
# 				word_emb[word] = np.array(list(map(float,vec.split())))
# 	word_emb_size = len(word_emb)
# 	print("Found "+str(word_emb_size)+" words in glove, out of "+str(vocab_size)+" words in vocabulary")
# 	return word_emb

#WORD_EMB_DIM = 300
SENT_DIM = 300
NUM_CLASSES = 2
#data_file = "dlnd_ablation1_data.p"
# if not os.path.exists(data_file):
#glove_path = "../../glove.840B.300d.txt"
dlnd_path = "new_dlnd/"
all_direc = [dlnd_path+direc+"/"+direc1+"/" for direc in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+direc) for direc1 in os.listdir(dlnd_path+direc)]
source_files = [[direc+"source/"+file for file in os.listdir(direc+"source/") if file.endswith(".txt")] for direc in all_direc]
target_files = [[direc+"target/"+file for file in os.listdir(direc+"target/") if file.endswith(".txt")] for direc in all_direc]
sources = [[sent for file in source_files[i] for sent in parse_file(file)] for i in range(len(target_files)) for j in range(len(target_files[i]))]
targets = [parse_file(target_files[i][j]) for i in range(len(target_files)) for j in range(len(target_files[i]))]
gold = [1 if [child.get("DLA") for child in ET.parse(file[:-4]+".xml").getroot() if "DLA" in child.attrib][0]=="Novel" else 0 for direc in target_files for file in direc]
# 	word_emb = fetch_glove_embeddings(sources+targets,glove_path)
# 	pickle.dump([sources,targets,gold,word_emb],open(data_file,"wb"))
# else:
# 	sources,targets,gold,word_emb = pickle.load(open(data_file,"rb"))
#max_sents= max([len(doc) for doc in sources+targets])
max_sents = 98
print("Max sentences in a document: "+str(max_sents))
# max_words = max([len(sent) for doc in sources+targets for sent in doc])
# max_words = 157
# print("Max words in a sentence: "+str(max_words))
print("Total target documents: "+str(len(targets)))
# vocab_size = len(word_emb)
# emb_mat = np.zeros((vocab_size+1,WORD_EMB_DIM),dtype="float32")
# vocab = dict()
# emb_mat[0] = np.zeros((WORD_EMB_DIM),dtype="float32")
# for i,word in enumerate(word_emb):
# 	emb_mat[i+1] = word_emb[word]
# 	vocab[word] = i+1
gold_list = np.array([i for i in gold])
sources = np.array(sources)
targets = np.array(targets)

kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 8309)
save_file = "dlnd_ablation1_ten_fold_progress.p"
fold = 1
if os.path.exists(save_file):
	predictions,golds,folds_complete,target,source,attentions = pickle.load(open(save_file,"rb"))
else:
	predictions = []
	golds = []
	folds_complete = 0
	target = []
	source = []
	attentions = []

for train,test in kfold.split(np.zeros(len(gold_list)),gold_list):
	if fold <= folds_complete:
		fold+=1
		continue
	print("\nFold: "+str(fold))
	fold+=1

	print("\nCompiling model\n")
	# doc_encode = Bidirectional(LSTM(SENT_DIM/2, return_sequences=True,dropout_W=0.0, dropout_U=0.0),input_shape=(max_sents, SENT_DIM))
 	#doc_encode = _BiRNNEncoding(max_sents, SENT_DIM, dropout=0.2)
	tgt_in = Input(shape=(max_sents,SENT_DIM), dtype='float32')
	srcs_in = Input(shape=(max_sents,SENT_DIM), dtype='float32')
	#tgt_vec = doc_encode(tgt_in)
	#src_vec = doc_encode(srcs_in)
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
	
	cb = [ModelCheckpoint("temp2_model.hdf5",monitor="val_loss",verbose=1,save_best_only=True,save_weights_only=True)]

	print("\nTraining model\n")
	history = model.fit(x=[train_tgt_vec,train_source_vec],y=train_gold,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,verbose=1,validation_split=0.1,shuffle=True,callbacks=cb)
	model.load_weights("temp2_model.hdf5")

	preds = model.predict(x=[test_tgt_vec,test_source_vec],batch_size=BATCH_SIZE,verbose=1)
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
	pickle.dump([predictions,golds,fold-1,target,source,attentions],open(save_file,"wb"))

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
