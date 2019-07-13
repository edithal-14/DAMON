# config variables
gpu_id = 2
dlnd_path = "TAP-DLND-1.0_LREC2018_modified"
glove_path = "glove.840B.300d.txt"
encoder_dir = "sentence_encoder"
encoder_path = encoder_dir + "/encoder/model_2048_attn.pickle"
save_file_path = "dlnd_data_attn.p"

import torch
torch.cuda.set_device(gpu_id)
import pickle
import os
import nltk
import xml.etree.ElementTree as ET
import numpy as np
import sys
sys.path.append(encoder_dir)

#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
topics  = [dlnd_path+"/"+dir+"/"+subdir for dir in os.listdir(dlnd_path) if os.path.isdir(dlnd_path+"/"+dir) for subdir in os.listdir(dlnd_path+"/"+dir)]
targets = list()
sources = list()
gold = list()
for topic in topics:
	src = [open(topic+"/source/"+doc,"r").read().decode("utf-8","ignore") for doc in os.listdir(topic+"/source") if doc.endswith(".txt")]	
	src = nltk.sent_tokenize(" . ".join(src))
	for doc in os.listdir(topic+"/target"):
		if doc.endswith(".txt"):
			targets.append(nltk.sent_tokenize(open(topic+"/target/"+doc,"r").read().decode("utf-8","ignore")))
			sources.append(src)
			gold.append(1 if [tag.attrib["DLA"] for tag in ET.parse(topic+"/target/"+doc[:-4]+".xml").findall("feature") if "DLA" in tag.attrib.keys()][0]=="Novel" else 0)
target_sentences = [sent for doc in targets for sent in doc]
print('Loading encoder')
infersent = torch.load(encoder_path)
print('Loading glove vectors')
infersent.set_glove_path(glove_path)
print('Building vocab')
infersent.build_vocab(target_sentences,tokenize=True)
print('Encoding target sentences')
all_vecs = infersent.encode(target_sentences,tokenize=True)
print('Encoding complete!')
target_vecs = []
i=0
for doc in targets:
	target_vecs.append(np.array(all_vecs[i:i+len(doc)]))
	i+=len(doc)

source_sentences = [sent for doc in sources for sent in doc]
print('Loading encoder')
infersent = torch.load(encoder_path)
print('Loading glove vectors')
infersent.set_glove_path(glove_path)
print('Building vocab')
infersent.build_vocab(source_sentences,tokenize=True)
print('Encoding source sentences')
all_vecs = infersent.encode(source_sentences,tokenize=True)
print('Encoding complete')
source_vecs = []
i = 0
for doc in sources:
	source_vecs.append(np.array(all_vecs[i:i+len(doc)]))
	i+=len(doc)

print('Dumping data')
pickle.dump([targets,sources,target_vecs,source_vecs,gold],open(save_file_path,"wb"))
