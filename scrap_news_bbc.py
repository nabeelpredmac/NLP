#f="E:\\Studies\\python\\bbc\\business"
#import os

'''
details : scrapping the news data from downloaded bbc news dataset and then clean it

'''

import pandas as pd


import glob
import errno


def read_txt(files,cat):

    l2=list()
    for name in files:
        try:
            with open(name) as f:
                l3=f.readlines()#.replace('\n', '')
                l3 = ','.join(map(str, l3))
                l3 =l3.replace('\n', '')
                l2.append(l3)
                
        except IOError as exc: #Not sure what error this is
            if exc.errno != errno.EISDIR:
                raise
    news_df = pd.DataFrame(l2,columns=['description'])
    news_df['category'] = cat
    return news_df 

path1 = 'E:\\Studies\\python\\bbc\\business\\*.txt' #note C:
path2 = 'E:\\Studies\\python\\bbc\\entertainment\\*.txt'
path3 = 'E:\\Studies\\python\\bbc\\politics\\*.txt'
path4 = 'E:\\Studies\\python\\bbc\\sport\\*.txt'
path5 = 'E:\\Studies\\python\\bbc\\tech\\*.txt'

files1 = glob.glob(path1)
files2 = glob.glob(path2)
files3 = glob.glob(path3)
files4 = glob.glob(path4)
files5 = glob.glob(path5)

files=[files1,files2,files3,files4,files5]
cats=['business','entertainment','politics','sport','tech']

news_df=pd.DataFrame()

for i in range(0,5):
    news_df=news_df.append(read_txt(files[i],cats[i]))

news_df.to_csv('E:/Studies/python/news_catagory/bbc_news.csv', encoding='utf-8', index=False)



