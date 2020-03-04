import pickle
import matplotlib.pyplot as plt
import os

#targets,sources,gold,preds,attention = pickle.load(open("sample_heatmap_data.p","rb"))
preds, gold, folds, targets, sources, attention, probs = pickle.load(open("../dlnd_decom_attn_ten_fold_progress.p","rb"))

# We have the results for n-fold cross validation
print('No. of cross validation folds: %d' % folds)

# Collect results of all the folds
targets  = [val for fold in targets for val in fold]
sources  = [val for fold in sources for val in fold]
attention  = [val for fold in attention for val in fold]

attention1 = []
#targets = targets.tolist()
#sources = sources.tolist()
for i in range(len(targets)):
	tgt_start = len([1 for sent in targets[i] if sent=="<PAD>"])
	targets[i] = targets[i][tgt_start:]
	src_start = len([1 for sent in sources[i] if sent=="<PAD>"])
	sources[i] = sources[i][src_start:]
	attention1.append(attention[i][tgt_start:,src_start:])
attention = attention1

target_dir = "dlnd_target_docs"
os.mkdir(target_dir)
for i in range(len(targets)):
	with open(target_dir+"/target"+str(i+1)+".txt","w") as f:
		f.write("\n-------------\n".join([str(j)+": "+targets[i][j].encode("ascii","ignore") for j in range(len(targets[i]))]))

source_dir = "dlnd_source_docs"
os.mkdir(source_dir)
for i in range(len(sources)):
	with open(source_dir+"/source"+str(i+1)+".txt","w") as f:
		f.write("\n-------------\n".join([str(j)+": "+sources[i][j].encode("ascii","ignore") for j in range(len(sources[i]))]))

images_dir = "heatmaps"
os.mkdir(images_dir)
for i in range(len(attention)):
	for j in range(attention[i].shape[0]):
		for k in range(attention[i].shape[1]):
			attention[i][j][k]/=10000
	title = ""
	if gold[i]==1:
		title = title+"Actual: Novel        "
	else:
		title = title+"Actual: Non-Novel	"
	if preds[i]==1:
		title = title+"Predicted: Novel"
	else:
		title = title+"Predicted: Non-Novel "
	plt.title(title)
	plt.xlabel("Source document sentences")
	plt.ylabel("Target document sentences")
	#for j in range(attention[i].shape[0]):
	#	for k in range(attention[i].shape[1]):
	#		plt.text(k+0.5,j+0.5,'.%3f' % attention[i][j][k],horizontalalignment='center',verticalalignment='center')
	heatmap = plt.pcolor(attention[i])
	plt.colorbar(heatmap)
	plt.savefig(images_dir+"/heatmap_"+str(i+1)+".png")
	plt.close()
