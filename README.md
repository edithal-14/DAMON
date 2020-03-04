# DAMON
Document Attention MOdel for Novelty (DAMON) a technique to identify redundant documents.

This is the codebase used in the writing of the paper called: **"Is Your Document Novel? Let Attention Guide You. An Attention-Based Model For Document Level Novelty Detection"** which is under review in the Natural Langauge Engineering (NLE) journal.


## Requirements
- Tensorflow
- Keras
- PyTorch
- Matplotlib
- GLoVe common crawl pre-trained word vectors (http://nlp.stanford.edu/data/glove.840B.300d.zip)
- TAP-DLND-1.0 (https://www.iitp.ac.in/~ai-nlp-ml/resources.html#)

## Directories
- dlnd: Contains various scripts and the DLND dataset in .zip format
    - scripts: Contains various scripts used for obtaining results in the paper
- apwsj: Contains .tar file containing judgements on APWSJ dataset and other .ttxt files derived from it
    - scripts: Contains various scripts used for obtaining results in the paper
- sentence_encoder: Contains scripts from Infersent used for generating sentence embeddings
