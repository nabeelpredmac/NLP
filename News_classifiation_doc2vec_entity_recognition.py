# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:33:47 2018

DEscription :
    Finding the news category using the bbc news dataset 
    category as general,politics,sports ,...etc
	used doc2vec and entity recognition

@author: Nabeel
"""


import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

def clean_sentance(st): 
   cst=re.sub(r'^[^0-9a-zA-Z]+', '', st).strip()
   cst=re.sub(r'[^0-9a-zA-Z.?\"\']+$', '', cst).strip()
   return(cst)

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

full_data=pd.read_csv('E:\\Studies\\python\\news_catagory\\train.csv',sep=',',encoding = 'ISO-8859-1')
full_data=full_data[['description','category']].drop_duplicates()
#full_data=full_data.sort_values(['level'],ascending=[1])



full_data = full_data.dropna()
#full_data=full_data.drop_duplicates()
description_data = (full_data.description)
#title_data = (full_data.title)


tokenizer = RegexpTokenizer(r'\w+')
#stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above


full_data.description=full_data.description.apply(clean_sentance)
#full_data.title=full_data.title.apply(clean_sentance)



##### Creating an class to return iterator object

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])
              
              
description_data = nlp_clean(description_data)
#title_data       =  nlp_clean(title_data)

data_len=list()
word_len_sum =list()

for i in range(0,len(description_data)):
    data_len.append(len(description_data[i]))
    sums=0
    for j in range(0,len(description_data[i])):
        sums = sums+len(description_data[i][j])
    
    word_len_sum.append(sums)    


docLabels=list(range(0, len(full_data)))
#iterator returned over all documents
it = LabeledLineSentence(description_data, docLabels)
#it_title = LabeledLineSentence(title_data, docLabels)

########################################################################
########  model for description2vec ###########################################
########################################################################

nbbc_model = gensim.models.Doc2Vec(size=400, min_count=0, alpha=0.025, min_alpha=0.025)
nbbc_model.build_vocab(it)
#training of model
for epoch in tqdm(range(10)):
 print ('iteration '+str(epoch+1))
 nbbc_model.train(it,total_examples=nbbc_model.corpus_count,epochs=nbbc_model.iter)
 nbbc_model.alpha -= 0.002
 nbbc_model.min_alpha = nbbc_model.alpha
 nbbc_model.train(it,total_examples=nbbc_model.corpus_count,epochs=nbbc_model.iter)
#saving the created model
nbbc_model.save('doc2vec.nbbc_model')
print ('model saved')

''' loading model '''
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.nbbc_model')

curr_vec = pd.DataFrame(list(d2v_model.docvecs))

########################################################################
########  model for title ###########################################
########################################################################
#
#tmodel = gensim.models.Doc2Vec(size=100, min_count=0, alpha=0.025, min_alpha=0.025)
#tmodel.build_vocab(it_title)
##training of model
#for epoch in range(50):
# print ('iteration '+str(epoch+1))
# tmodel.train(it,total_examples=tmodel.corpus_count,epochs=tmodel.iter)
# tmodel.alpha -= 0.002
# tmodel.min_alpha = nmodel.alpha
# tmodel.train(it,total_examples=tmodel.corpus_count,epochs=tmodel.iter)
##saving the created model
#tmodel.save('doc2vec.tmodel')
#print ('model saved')
#
#''' loading model '''
#t2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.tmodel')

###############################################################################
#
#title_vec = pd.DataFrame(list(t2v_model.docvecs))
#title_vec1=title_vec
#title_vec1.columns = [('w'+str(i)) for i in range(0,100)]
#
#curr_vec = pd.DataFrame(list(d2v_model.docvecs))
##start testing
##printing the vector of document at index 1 in docLabels
#''' words in vector form adding to the data is commented down '''
#curr_vec_merge = pd.concat([curr_vec.reset_index(drop=True), title_vec1], axis=1)
#temp = curr_vec
#curr_vec=curr_vec_merge

################################################################################
######## train and test data creation ##########################################
###############################################################################



#curr_vec['data_len']=list(data_len)
#curr_vec['word_len_sum']=list(word_len_sum)
#curr_vec['word']=full_data['word']

lvl=list(full_data['category'] )

curr_vec['category'] = lvl

from sklearn.model_selection import train_test_split


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


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.callbacks import TensorBoard
from time import time
from sklearn import preprocessing
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical

le=preprocessing.LabelEncoder()
le.fit(full_data['category'])

curr_vec1=curr_vec
curr_vec1['category'] =le.transform(list(curr_vec['category']))



train_x1, test_x1, train_y1, test_y1 = split_dataset(curr_vec1, 0.7, headers[0:-1], headers[-1])



# fix random seed for reproducibility
np.random.seed(7)


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
md.add(Dropout(0.5, input_shape=(400,)))
md.add(Dense(10, input_dim=400, activation='relu'))
md.add(Dense(10, activation='relu'))
md.add(Dense(4, activation='softmax'))
#

# Compile model
md.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

#from keras.callbacks import TensorBoard
run_name="news_cat_model_bbc4"
tensor_board_log_dir='E:\\Studies\\python\\news_catagory\\tf_logs\\'
#run_name="conv_8_3_layers"
tensorboard = TensorBoard(log_dir=tensor_board_log_dir+run_name,write_graph=True,embeddings_layer_names=None, embeddings_metadata=None)
r1 = md.fit(binary_train_x,  binary_train_y, epochs=50, batch_size=10, validation_data = (binary_test_x,binary_test_y),callbacks=[tensorboard] )
#

#md.evaluate(test_x1_t, test_y1_t, verbose=0)
###############################################################################
############ entity finding ###################################################
###############################################################################


###############################################################################

import re
import spacy
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.corpus import wordnet

#inputfile = open('inputfile.txt', 'r')
#String= full_data.description[1]
nlp = spacy.load('en_core_web_sm')
full_data['persons']=""
damn=list()

def candidate_name_extractor(input_string, nlp):
    input_string = str(input_string)

    doc = nlp(input_string)

    # Extract entities
    doc_entities = doc.ents

    # Subset to person type entities
    doc_persons = filter(lambda x: x.label_ == 'PERSON', doc_entities)
    doc_persons = filter(lambda x: len(x.text.strip().split()) >= 2, doc_persons)
    doc_persons = list(map(lambda x: x.text.strip(), doc_persons))
    print(doc_persons)
    # Assuming that the first Person entity with more than two tokens is the candidate's name
    #candidate_name = doc_persons[0]
    return doc_persons

#if __name__ == '__main__':
for i in range(0,len( full_data.description)):
    String = full_data.iloc[i,0]
    names = candidate_name_extractor(String, nlp)
    names = ','.join(map(str, names))
    names = names.replace("[", '')
    names = names.replace("]", '')
    names = names.replace("'", '')
    #full_data.iloc[i]['persons'] = names
    full_data.set_value(i, 'persons', names)

#full_data['persons']=full_data['persons'].replace('[', '')
#full_data['persons']=full_data['persons'].replace(']', '')
#full_data['persons']=full_data['persons'].replace("'", '')

full_data = full_data.dropna()
full_data.to_csv("E:\\Studies\\python\\news_catagory\\categorized_output.csv",index=False)
#print(names)


## <<<<<<<<<<<<<<<<<<<<<<<<<<< Real code ends here >>>>>>>>>>>>>>>>>>>>>>>> ##
    

################################################################################
######## Random Forest modelling ##############################################
###############################################################################



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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

train_x, test_x, train_y, test_y = split_dataset(curr_vec, 0.7, headers[0:-1], headers[-1])

random_model = RandomForestClassifier(n_estimators=400)#,verbose=0,n_jobs=-1, oob_score = True,
                                      #random_state =None)
random_model.fit(train_x, train_y)

results = random_model.predict(test_x)

result_predicted=list(results)
compare=pd.DataFrame(list(test_y),columns={'actual'} )
compare['predicted']=result_predicted

print ("Train Accuracy :: ", accuracy_score(train_y, random_model.predict(train_x)))
print ("Test Accuracy  :: ", accuracy_score(test_y, results))
print (" Confusion matrix \n", confusion_matrix(test_y, results))


###############################################################################
############### MLP Classifier ################################################
###############################################################################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(solver='adam',hidden_layer_sizes=(2,8,6), random_state=1
                          ,activation='relu')
mlp_model.fit(train_x, train_y)

mlp_results = mlp_model.predict(test_x)


mlp_result_predicted=list(mlp_results)
mlp_compare=pd.DataFrame(list(test_y),columns={'actual'} )
mlp_compare['predicted']=mlp_result_predicted

print ("Train Accuracy :: ", accuracy_score(train_y, mlp_model.predict(train_x)))
print ("Test Accuracy  :: ", accuracy_score(test_y, mlp_results))
print (" Confusion matrix \n", confusion_matrix(test_y, mlp_results))



    


#################################################################################
#############  test part #######################################################


#################################################################################
############## For scraped data ####################################################
#################################################################################
full_data_t=pd.read_csv('E:\\Studies\\python\\news_catagory\\news2.csv',sep=',',encoding = 'ISO-8859-1')
full_data_t=full_data_t[['description','category']]
full_data_t=full_data_t[['description','category']].drop_duplicates()

full_data_t['category'] = full_data_t['category'].map({'technology': 'tech', 'sports': 'sports','politics' : 'politics',
           'business' : 'business' , 'entertainment' : 'entertainment','general' : 'general', 'world' : 'world'})
#full_data_t=full_data_t[(full_data_t.category.str.contains('general')) & (full_data_t.category.str.contains('science'))]

full_data_t = full_data_t[full_data_t['category'].isin([ 'tech','business','sports','world'])]#,'entertainment','politics'])]
full_data_t = full_data_t.dropna()

full_data_t.description=full_data_t.description.apply(clean_sentance)

description_data_t = (full_data_t.description)

             
description_data_t = nlp_clean(description_data_t)


docLabels=list(range(0, len(full_data_t)))
#iterator returned over all documents
it_t = LabeledLineSentence(description_data_t, docLabels)


nbbc_model_t = gensim.models.Doc2Vec(size=400, min_count=0, alpha=0.025, min_alpha=0.025)
nbbc_model_t.build_vocab(it_t)
#training of model
for epoch in tqdm(range(10)):
 print ('iteration '+str(epoch+1))
 nbbc_model_t.train(it_t,total_examples=nbbc_model_t.corpus_count,epochs=nbbc_model_t.iter)
 nbbc_model_t.alpha -= 0.002
 nbbc_model_t.min_alpha = nbbc_model_t.alpha
 nbbc_model_t.train(it_t,total_examples=nbbc_model_t.corpus_count,epochs=nbbc_model_t.iter)
#saving the created model
nbbc_model_t.save('doc2vec.nbbc_model_t')
print ('model saved')

''' loading model '''
d2v_model_t = gensim.models.doc2vec.Doc2Vec.load('doc2vec.nbbc_model_t')

curr_vec_t = pd.DataFrame(list(d2v_model_t.docvecs))

lvl=list(full_data_t['category'] )

curr_vec_t['category'] = lvl


curr_vec2=curr_vec_t
curr_vec2['category'] =le.transform(list(curr_vec_t['category']))



train_x1_t, test_x1_t, train_y1_t, test_y1_t = split_dataset(curr_vec2, 1, headers[0:-1], headers[-1])


binary_test_x_t=test_x1_t.values
binary_test_y_t = to_categorical(test_y1_t)


md.evaluate(binary_test_x_t, binary_test_y_t, verbose=0)

####################################################################################


import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.corpus import wordnet

String =full_data.description[1]

Sentences = nltk.sent_tokenize(String)
Tokens = []
for Sent in Sentences:
    Tokens.append(nltk.word_tokenize(Sent)) 
Words_List = [nltk.pos_tag(Token) for Token in Tokens]

Nouns_List = []

for List in Words_List:
    for Word in List:
        if re.match('[NN.*]', Word[1]):
             Nouns_List.append(Word[0])

Names = []
for Nouns in Nouns_List:
    if not wordnet.synsets(Nouns):
        Names.append(Nouns)

print (Names)
