# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:29:10 2018

@author: Nabeel

Description : Here the sentence converted to vectorised form using word2vec model .
            Vocabulary for it is created before.
            And this word2vec model is saved .
"""

from string import punctuation
from os import listdir
from gensim.models import Word2Vec

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if (filename.startswith('readme')):
            continue
#		if not is_trian and not filename.startswith('cv9'):
#			continue
		# create the full path of the file to open
        path = directory + '/' + filename
		# load and clean the doc
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)
        # add lines to list
        lines += doc_lines
    return lines

# load the vocabulary
vocab_filename = 'E:/Studies/python/sentimental/vocab1.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load training data

# load all training reviews
all_docs = process_docs('E:/Studies/python/sentimental/sentiment labelled sentences', vocab, True)
#negative_docs = process_docs('E:/Studies/python/sentimental/review_polarity/txt_sentoken/neg', vocab, True)
#train_docs = all_docs[0:2100]
sentences = all_docs[0:2100]
print('Total training sentences: %d' % len(sentences))

# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'E:/Studies/python/sentimental/embedding_word2vec1.txt'
model.wv.save_word2vec_format(filename, binary=False)