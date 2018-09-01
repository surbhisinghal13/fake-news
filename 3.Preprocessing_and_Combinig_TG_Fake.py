######################################################################
##  Script Info: It preprocessses, cleans, lemmatizes the news articles  
######################################################################

import pandas as pd
from collections import Counter
import re
import numpy as np
import nltk
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn import feature_extraction


_wnl = nltk.WordNetLemmatizer()

STOP_LIST=['guardian','theguardian']

def lemmatization(w):
    return _wnl.lemmatize(w).lower()

#Function to remove Non Alphanumeric Characters#######

def clean(s):
    return " ".join(re.findall(r'\w+', str(s), flags=re.UNICODE)).lower()


def Clean_stopwords(l):
    # Removes stopwords from a list of tokens
    return [ lemmatization(w) for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS and w not in STOP_LIST and len(lemmatization(w)) > 1]
    

DataFrameTG=pd.read_csv("./tempdata/Clean_TheGuardian.csv")

DataFrameTG["fakeness"] = 0  
DataFrameFake = pd.read_csv("./tempdata/fake.csv")

#print("The columns of the guardians are :",DataFrameFake.columns)

DataFrameTG = DataFrameTG.rename(columns={'bodyText' : 'body'})

DataFrameFake = DataFrameFake.rename(columns={'text':'body','title':'headline'})

# Dropping unnecesary columns
DataFrameFake.drop([ u'uuid',u'ord_in_thread',u'author',u'published', 
         u'language', u'crawled', u'site_url', u'country',
        u'thread_title', u'spam_score', u'replies_count', u'participants_count',
        u'likes', u'comments', u'shares', u'type', u'domain_rank',u'main_img_url'],inplace=True,axis=1)

DataFrameTG.drop([u'Unnamed: 0',  u'apiUrl', u'fields',u'id', 
        u'isHosted', u'sectionId', u'sectionName', u'type',
         u'webTitle', u'webUrl', u'pillarId', u'pillarName',u'webPublicationDate'],inplace=True,axis=1)
         
#print("The columns of the guardians are :",DataFrameTG.columns,"Size",len(DataFrameTG))
DataFrameFake["fakeness"] = 1

# Concta the DataFrames of fakeNews and TheGuardian
DataFrameComplete = DataFrameFake.append(DataFrameTG, ignore_index=True)


for index, row in DataFrameComplete.iterrows():
    clean_headline=clean(row['headline'])
    temp_headline=" ".join(Clean_stopwords(clean_headline.split()))
    
    DataFrameComplete.set_value(index,'headline',temp_headline)
    
    clean_body=clean(row['body'])
    temp_body=" ".join(Clean_stopwords(clean_body.split()))
    DataFrameComplete.set_value(index,'body',temp_body)
    

DataFrameComplete = DataFrameComplete[DataFrameComplete.body != ""]
DataFrameComplete = DataFrameComplete[DataFrameComplete.headline != ""]

#Dropping the Nan values and info
DataFrameComplete=DataFrameComplete.dropna()
DataFrameComplete=DataFrameComplete[DataFrameComplete.headline!="nan"]
DataFrameComplete=DataFrameComplete[DataFrameComplete.body!="nan"]


DataFrameComplete=DataFrameComplete[['body','headline','fakeness']]
DataFrameComplete=DataFrameComplete.sample(frac=1).reset_index(drop=True)
DataFrameComplete.to_csv("./tempdata/Complete_DataSet.csv")

