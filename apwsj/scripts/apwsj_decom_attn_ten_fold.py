import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from spacy_decomposable_attention import _BiRNNEncoding
from spacy_decomposable_attention import _Attention
from spacy_decomposable_attention import _SoftAlignment
from spacy_decomposable_attention import _Comparison
from spacy_decomposable_attention import _Entailment
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras import backend as K

def pad_or_truncate(docs,max_sents):
	ans = [["<PAD>" for j in range(max_sents)] for i in range(len(docs))]
	for i in range(len(docs)):
		docs[i] = docs[i][:max_sents]
		for k in range(len(docs[i])):
			ans[i][max_sents-len(docs[i])+k] = docs[i][k]
	ans = np.array(ans)
	return ans

def pad_or_truncate1(mats,max_sents):
	global SENT_DIM
	ans = np.zeros((len(mats),max_sents,SENT_DIM),dtype='float32')
	for i in range(len(mats)):
		mats[i] = mats[i][:max_sents]
		ans[i][max_sents-mats[i].shape[0]:] = mats[i]
	return ans

SENT_DIM = 2048
NUM_CLASSES = 2
targets,sources,target_vecs,source_vecs,gold = pickle.load(open("apwsj_data_maxpool_snli.p","rb"))
# max_sents= max([len(doc) for doc in sources+targets])
max_sents = 336
print("Max sentences: "+str(max_sents))

print("Total no. of instances: "+str(len(gold)))
print("Total no. of Novel instances: "+str(sum(gold)))
print("Total no. of Non-Novel instances: "+str(len(gold)-sum(gold)))
print("Percentage of Novel instances: "+str((sum(gold)/float(len(gold)))*100)+"%")

targets = pad_or_truncate(targets,max_sents)
sources = pad_or_truncate(sources,max_sents)
target_vecs = pad_or_truncate1(target_vecs,max_sents)
source_vecs = pad_or_truncate1(source_vecs,max_sents)
gold_list = [i for i in gold]
gold = to_categorical(gold,num_classes=NUM_CLASSES)

kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 9274)
save_file = "apwsj_decom_attn_maxpool_snli_ten_fold_progress.p"
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
	tgt = Input(shape=(max_sents,SENT_DIM), dtype='float32')
	srcs = Input(shape=(max_sents,SENT_DIM), dtype='float32')
	attention = _Attention(max_sents,SENT_DIM,dropout=0.2)(tgt,srcs)
	align1= _SoftAlignment(max_sents,SENT_DIM)(srcs,attention)
	align2= _SoftAlignment(max_sents,SENT_DIM)(tgt,attention,transpose=True)
	vec_l = _Comparison(max_sents,SENT_DIM,dropout=0.2)(tgt,align1)
	vec_r = _Comparison(max_sents,SENT_DIM,dropout=0.2)(srcs,align2)
	pds = _Entailment(SENT_DIM,NUM_CLASSES,dropout=0.2)(vec_l,vec_r)
	model = Model(inputs=[tgt,srcs],outputs=pds)
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.001),metrics=["accuracy"])

	cb = [ModelCheckpoint("temp2_model.hdf5",monitor="val_loss",verbose=1,save_weights_only=True,save_best_only=True)]

	get_attention_matrix = K.function([model.layers[0].input,model.layers[1].input,K.learning_phase()],[model.layers[3].output])

	NUM_EPOCHS = 20
	BATCH_SIZE = 25

	print("\nTraining model\n")
	history = model.fit(x=[target_vecs[train],source_vecs[train]],y=gold[train],batch_size=BATCH_SIZE,validation_split=0.1,epochs=NUM_EPOCHS,shuffle=True,verbose=1,callbacks=cb)
	model.load_weights("temp2_model.hdf5")

	preds = model.predict([target_vecs[test],source_vecs[test]],verbose=1,batch_size=BATCH_SIZE)
	probs.append(preds)
	preds = np.argmax(preds,axis=1)
	gold_test = np.argmax(gold[test],axis=1)
	predictions.append(preds)
	golds.append(gold_test)
	target.append(targets[test])
	source.append(sources[test])
	att = np.array([get_attention_matrix([target_vecs[test][st_index:st_index+BATCH_SIZE],source_vecs[test][st_index:st_index+BATCH_SIZE],0])[0] for st_index in range(0,target_vecs[test].shape[0],BATCH_SIZE)])
	attentions.append(att)
	test_acc = accuracy_score(gold_test,preds)
	p,r,f,_ = precision_recall_fscore_support(gold_test,preds)
	print("Testing accuracy: "+str(test_acc))
	print("Confusion matrix: \n"+str(confusion_matrix(gold_test,preds,labels=[0,1])))
	print("Precision: "+str(p))
	print("Recall: "+str(r))
	print("F-score: "+str(f))
	# os.remove(save_file)
	# pickle.dump([predictions,golds,fold-1,target,source,attentions,probs],open(save_file,"wb"))

predictions = [i for l in predictions for i in l]
golds = [i for l in golds for i in l]
test_acc = accuracy_score(golds,predictions)
p,r,f,_ = precision_recall_fscore_support(golds,predictions)
print("Testing accuracy: "+str(test_acc))
print("Confusion matrix: \n"+str(confusion_matrix(golds,predictions,labels=[0,1])))
print("Precision: "+str(p))
print("Recall: "+str(r))
print("F-score: "+str(f))
pickle.dump([predictions,golds,fold-1,target,source,attentions,probs],open(save_file,"wb"))