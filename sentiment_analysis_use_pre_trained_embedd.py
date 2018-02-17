# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:35:53 2018

@author: Nabeel

Description  : 
                A convolution neural network model that used to find whether the given
                review is +ve or -ve ,
                Here the pre-trained word2vec model (previously created by us) is used 
                to convert the sentence ,and create the weight matrix using it. 
                Then embedding this weight matrix with the cnn model . 
"""

from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    
    text_list = list()
	# walk through all files in the folder
    for filename in listdir(directory):
		# skip any reviews in the test set
        if  filename.startswith('readme'):
            continue
#		if not is_trian and not filename.startswith('cv9'):
#			continue
		# create the full path of the file to open
        path = directory + '/' + filename
		# load the doc
        
        with open(path) as f:
            text1 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        text1 = [x.strip() for x in text1]
        
        for i in range(0,len(text1)):
            text2 = (text1[i][0:-1])
            text2 = text2.replace('\t','')
            text3 = clean_doc(text2, vocab)
            text_list.append(text3)
        
    return text_list

# load all output -ve / +ve
def process_docs1(directory):
    
    outs =list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('readme'):
            
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        
        #text1 = (file.read() )   
        with open(path) as f:
            text1 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        text1 = [x.strip() for x in text1]
        
        for i in range(0,len(text1)):
            outs.append(text1[i][-1])
    return outs


# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

# load the vocabulary
vocab_filename = 'E:/Studies/python/sentimental/vocab1.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


# load all training reviews
all_docs = process_docs('E:/Studies/python/sentimental/sentiment labelled sentences', vocab, True)
#negative_docs = process_docs('E:/Studies/python/sentimental/review_polarity/txt_sentoken/neg', vocab, True)
train_docs = all_docs[0:2100]
outputs = process_docs1('E:/Studies/python/sentimental/sentiment labelled sentences')

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array(outputs[0:2100])

# load all test reviews

test_docs = all_docs[2101:3000]
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array(outputs[2101:3000])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('E:/Studies/python/sentimental/embedding_word2vec1.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))