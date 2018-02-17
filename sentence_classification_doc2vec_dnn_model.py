
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:59:39 2018

@author: nabeel


Desription  : 
    
    sentence Classification using  gensim doc2vec vectorised senteses, modelled using deep learning neural network . 
    Keras with Tensor Flow backend :
	

"""
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import re

full_data=pd.read_csv('E:\\Projects\\Python works\\nlp_grammer\\sent_2_vec\\Full_vocabulary_cleaned_new.csv',sep=',',
                      encoding = 'ISO-8859-1')
full_data=full_data[['sentences','level','word']].drop_duplicates()
#full_data=full_data.sort_values(['level'],ascending=[1])

full_data = full_data.dropna()
#full_data=full_data.drop_duplicates()
data = (full_data.sentences)
#full_data['word_no']=0

data_word= full_data.word


##### For cleaning the data 

tokenizer = RegexpTokenizer(r'\w+')
#stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above
def clean_sentance(st): 
   cst=re.sub(r'^[^0-9a-zA-Z]+', '', st).strip()
   cst=re.sub(r'[^0-9a-zA-Z.?\"\']+$', '', cst).strip()
   return(cst)

full_data.sentences=full_data.sentences.apply(clean_sentance)

def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      #dlist=re.sub(r'^[^0-9a-zA-Z]+', '', d).strip()
      #dlist=re.sub(r'[^0-9a-zA-Z.?\"\']+$', '', d).strip()
       #return(cst)
      dlist = tokenizer.tokenize(new_str)
      #dlist = list(set(dlist).difference(stopword_set))
      dlist = list(dlist)
      new_data.append(dlist)
   return new_data

##### Creating an class to return iterator object

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])
              
              
data = nlp_clean(data)
data_word = nlp_clean(data_word)

data_len=list()
word_len_sum =list()

for i in range(0,len(data)):
    data_len.append(len(data[i]))
    sums=0
    for j in range(0,len(data[i])):
        sums = sums+len(data[i][j])
    
    word_len_sum.append(sums)    


docLabels=list(range(0, len(full_data)))
#iterator returned over all documents
it = LabeledLineSentence(data, docLabels)
it_word =  LabeledLineSentence(data_word, docLabels)
#sentence = gensim.models.doc2vec.LabeledSentence(words=[u'some', u'words', u'here'],labels= [u'SENT_1'])

model = gensim.models.Doc2Vec(size=400, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#training of model
for epoch in range(50):
 print ('iteration '+str(epoch+1))
 model.train(it,total_examples=model.corpus_count,epochs=model.iter)
 model.alpha -= 0.002
 model.min_alpha = model.alpha
 model.train(it,total_examples=model.corpus_count,epochs=model.iter)
#saving the created model
model.save('doc2vec.model')
print ('model saved')

#### Word 2 vector 

# Load Google's pre-trained Word2Vec model.
#model_word_google = gensim.models.KeyedVectors.load_word2vec_format

#model_word_google=gensim.models.KeyedVectors.load_word2vec_format('E:\\Projects\\Python works\\nlp_grammer\\sent_2_vec\\GoogleNews-vectors-negative300.bin', binary=True)  
''' loading words as list'''
data_word = list(full_data.word)

#data_word_vec = model_word_google.most_similar(data_word)

##### word2vec conversion model creation ###########

model_word = gensim.models.Doc2Vec(size=30, min_count=0, alpha=0.025, min_alpha=0.025)
#model_word.build_vocab(it_word)
#training of model
for epoch in range(50):
 print ('iteration '+str(epoch+1))
 model_word.train(it_word,total_examples=model_word.corpus_count,epochs=model_word.iter)
 model_word.alpha -= 0.002
 model_word.min_alpha = model_word.alpha
 model_word.train(it,total_examples=model_word.corpus_count,epochs=model_word.iter)
#saving the created model
model_word.save('doc2vec.model_word')
print ('word model saved')

#loading the models
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

w2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model_word')

word_vec = pd.DataFrame(list(w2v_model.docvecs))
word_vec1=word_vec
word_vec.columns = [('w'+str(i)) for i in range(0,30)]

curr_vec = pd.DataFrame(list(d2v_model.docvecs))
#start testing
#printing the vector of document at index 1 in docLabels
''' words in vector form adding to the data is commented down '''
#curr_vec_merge = pd.concat([curr_vec.reset_index(drop=True), word_vec], axis=1)
#temp = curr_vec
#curr_vec=curr_vec_merge

################################################################################
######## train and test data creation ##########################################
###############################################################################



curr_vec['data_len']=list(data_len)
#curr_vec['word_len_sum']=list(word_len_sum)
#curr_vec['word']=full_data['word']

lvl=list(full_data['level'] )

curr_vec['level'] = lvl

#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

headers=list(curr_vec)


#################################################################################
################ Using keras nn model ###########################################
################################################################################



from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(full_data['level'])

curr_vec1=curr_vec
curr_vec1['level'] =le.transform(list(curr_vec['level']))



train_x1, test_x1, train_y1, test_y1 = split_dataset(curr_vec1, 0.7, headers[0:-1], headers[-1])


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import TensorBoard
from time import time
# fix random seed for reproducibility
np.random.seed(7)


#
#from keras.layers import  Embedding
#from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
#

'''here the data modification'''


binary_train_x = train_x1.values
binary_train_y = to_categorical(train_y1)

binary_test_x=test_x1.values
binary_test_y = to_categorical(test_y1)

binary_train_x[0].shape
binary_test_x[0].shape

binary_train_y[0]
binary_test_y[0]

#

# create model
md = Sequential()
md.add(Dense(8, input_dim=401, activation='relu'))
md.add(Dense(8, activation='relu'))
md.add(Dense(6, activation='softmax'))
#

# Compile model
md.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

#from keras.callbacks import TensorBoard
run_name="nn_with_word_sum1"
tensor_board_log_dir='E:\\Projects\\Python works\\nlp_grammer\\sent_2_vec\\nabeel_Test\\tensor_board_log\\'
#run_name="conv_8_3_layers"
tensorboard = TensorBoard(log_dir=tensor_board_log_dir+run_name,write_graph=True,embeddings_layer_names=None, embeddings_metadata=None)


#tensorboard = TensorBoard(log_dir=tensor_board_log_dir+run_name,write_graph=True,embeddings_layer_names=None, embeddings_metadata=None)

#tensorboard = TensorBoard(log_dir="E:\\Projects\\Python works\\nlp_grammer\\sent_2_vec\\Tensor_board_log".format(time(),write_graph=True,embeddings_layer_names=None, embeddings_metadata=None))
r1 = md.fit(binary_train_x,  binary_train_y, epochs=5, batch_size=10, validation_data = (binary_test_x,binary_test_y),callbacks=[tensorboard] )
#





################ Main code ends here ###########################################







pp=md.predict(binary_test_x)

tests=pd.DataFrame(pp)

pred_list=list()
for i in range(0,len(tests)):
    pred_list.append((tests == max(tests.iloc[i,:])).idxmax(axis=1)[i])

compare_nn = pd.DataFrame(pred_list , columns={'predicted'})
compare_nn['real']= list(test_y1)

score = md.evaluate(binary_test_x, binary_test_y, batch_size=128)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#print ("Train Accuracy :: ", accuracy_score(test_y1, pred_list))
print ("Test Accuracy  :: ", accuracy_score(test_y1, pred_list))
print (" Confusion matrix \n", confusion_matrix(test_y1, pred_list))


import matplotlib.pyplot as plt

history=r1
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()