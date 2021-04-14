# word2vec
Homework2 of NLP: Word2Vec

## make dataset
`python dataset.py` for demo dataset

## training
`python train.py --embedding_dim 300 --wordnet_freq 0` for Word2Vec

`python train.py --embedding_dim 300 --wordnet_freq 5` for Word2Vec using WordNet at a frequency of 5.

All the results (loss and testing score) will be stored in the directory.

## testing
`python eval.py --load_path model_path`