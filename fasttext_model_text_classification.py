#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:38:21 2017

@author: nabeel
     test classification with fast text
"""

import fasttext

# Skipgram model
model = fasttext.skipgram('data.txt', 'model')
print model.words # list of words in dictionary

# CBOW model
model = fasttext.cbow('data.txt', 'model')
print model.words # list of words in dictionary

print model['king'] # get the vector of the word 'king'
model = fasttext.load_model('model.bin')
print model.words # list of words in dictionary
print model['king'] # get the vector of the word 'king'
classifier = fasttext.supervised('data.train.txt', 'model')
classifier = fasttext.supervised('data.train.txt', 'model', label_prefix='__label__')
result = classifier.test('test.txt')
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples
texts = ['example very long text 1', 'example very longtext 2']
#predicting with probability
labels = classifier.predict(texts)
print labels

# Or with the probability
labels = classifier.predict_proba(texts)
print labels
labels = classifier.predict(texts, k=3)
print labels

# Or with the probability
labels = classifier.predict_proba(texts, k=3)
print labels
model = fasttext.skipgram('train.txt', 'model', lr=0.1, dim=300)
#input_file     training file path (required)
#output         output file path (required)
#lr             learning rate [0.05]
#lr_update_rate change the rate of updates for the learning rate [100]
#dim            size of word vectors [100]
#ws             size of the context window [5]
#epoch          number of epochs [5]
#min_count      minimal number of word occurences [5]
#neg            number of negatives sampled [5]
#word_ngrams    max length of word ngram [1]
#loss           loss function {ns, hs, softmax} [ns]
#bucket         number of buckets [2000000]
#minn           min length of char ngram [3]
#maxn           max length of char ngram [6]
#thread         number of threads [12]
#t              sampling threshold [0.0001]
#silent         disable the log output from the C++ extension [1]
#encoding       specify input_file encoding [utf-8]

model = fasttext.cbow('train.txt', 'model', lr=0.1, dim=300)
model = fasttext.load_model('model.bin', encoding='utf-8')
#model.model_name       # Model name
#model.words            # List of words in the dictionary
#model.dim              # Size of word vector
#model.ws               # Size of context window
#model.epoch            # Number of epochs
#model.min_count        # Minimal number of word occurences
#model.neg              # Number of negative sampled
#model.word_ngrams      # Max length of word ngram
#model.loss_name        # Loss function name
#model.bucket           # Number of buckets
#model.minn             # Min length of char ngram
#model.maxn             # Max length of char ngram
#model.lr_update_rate   # Rate of updates for the learning rate
#model.t                # Value of sampling threshold
#model.encoding         # Encoding of the model
#model[word]            # Get the vector of specified word
classifier = fasttext.supervised(params)
classifier = fasttext.supervised('train.txt', 'model', label_prefix='__myprefix__',
                                 thread=4)
classifier = fasttext.load_model('classifier.bin', label_prefix='some_prefix')
labels = classifier.predict(texts, k)

# Or with probability
labels = classifier.predict_proba(texts, k)
