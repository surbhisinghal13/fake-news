import string
import numpy as np
import re
import nltk
#nltk.download()
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def count_2grams(raw):
    s=[]
    tokens = nltk.word_tokenize(raw)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    bigram=ngrams(nouns,2)
    s=list(bigram)
    
    return str(s)

def count_1grams(raw):
    s=[]
    tokens = nltk.word_tokenize(raw)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    unigram=ngrams(nouns,1)
    s=list(unigram)
    
    return str(s)
 
def count_3grams(raw):
    s=[]
    tokens = nltk.word_tokenize(raw)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    trigram=ngrams(nouns,3)
    s=list(trigram)
    return str(s)

def freq2(raw):
    string1=count_2grams(raw);
    list1=string1.split(' ')
    freq=0
    for grams in list1:
       freq +=1
    return freq

def freq1(raw):
    string1=count_1grams(raw);
    list1=string1.split(' ')
    freq=0
    for grams in list1:
       freq +=1
    return freq

def freq3(raw):
    string1=count_3grams(raw);
    list1=string1.split(' ')
    freq=0
    for grams in list1:
       freq +=1
    return freq

def tfidf(corpus):
    
    tfidf_vect=TfidfVectorizer(min_df=1, stop_words='english')
    tfidf_matrix=tfidf_vect.fit_transform(corpus)
    print(tfidf_matrix.todense())	

def extract_adjective(sentences):
    adj_sentences = list()
    count=0;
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        adj_tags = nltk.pos_tag(words)
        one_adj_sentence = ""
        for index, tag in enumerate(adj_tags, start = 0):
            one_tag = tag[1]
            if one_tag in ['JJ', 'JJR', 'JJS']:
                one_adj_sentence += words[index]
                one_adj_sentence += " "
                count+=1
        adj_sentences.append(one_adj_sentence)
       
    return count

def removePunc(input):
    '''
    :param input: string
    :return: string, without the punctuations
    '''
    #return input.translate(string.maketrans("",""), string.punctuation)
    return re.sub("[\.\t\,\:;\(\)\.]", "", input, 0, 0)

def numOfWords(input):
    '''
    :param input: string
    :return: number of words, number of continuous space
    '''
    splitted = input.split(" ")
    res=0
    for i in splitted:
        if len(i)>0:
            res+=1
    return res

def numOfChar(input):
    '''
    :param input: string
    :return: number of char
    '''
    return len(input)

def numOfPunc(input):
    '''
    :param input: string
    :return: number of punctuations
    '''
    return len(input)-len(removePunc(input))

def numOfContPunc(input):
    res=0;
    state=False
    for i in range(1,len(input)):
        if input[i] in string.punctuation:
            if input[i-1] in string.punctuation:
                if state:
                    pass
                else:
                    state=True
                    res+=1
            else:
                state=False
                pass
        else:
            state=False
    return res

def numOfContUpperCase(input):
    res = 0;
    state = False
    for i in range(1, len(input)):
        if input[i].isupper():
            if input[i - 1].isupper():
                if state:
                    pass
                else:
                    state = True
                    res += 1
            else:
                state = False
                pass
        else:
            state = False
    return res
    pass

def constructMat(data,label):
    '''
    :param file: input file
    :param label: the label of the data in the file
    :return: ndarray
    '''
    res=np.array([])
    line1=True
    i=0
    for line in data:
        if line1:
            line1=False
            cleaned = line.lower().strip()
            original = line.strip()
            fea1 = numOfWords(cleaned)
            fea2 = numOfChar(cleaned)
            fea3 = numOfPunc(cleaned)
            fea4 = numOfContPunc(cleaned)
            fea5 = numOfContUpperCase(original)
            fea8 = freq2(cleaned) 
                
            fea10 = freq1(cleaned)
                
            fea12 = freq3(cleaned)
               
            res1= np.array([[fea1, fea2, fea3, fea4, fea5,fea8,fea10,fea12,label[i]]])
            
        else:
            cleaned = line.lower().strip()
            original = line.strip()
            fea1 = numOfWords(cleaned)
            fea2 = numOfChar(cleaned)
            fea3 = numOfPunc(cleaned)
            fea4 = numOfContPunc(cleaned)
            fea5 = numOfContUpperCase(original)
                
            fea8 = freq2(cleaned)
              
            fea10 = freq1(cleaned)
                
            fea12 = freq3(cleaned)
                
            newrow1= np.array([[fea1, fea2, fea3, fea4, fea5,fea8,fea10,fea12,label[i]]])
            newrow = np.array([[fea1, fea2, fea3, fea4, fea5, label[i]]])
            
            res1= np.append(res1, newrow1, axis=0)
        i=i+1
    
    return res1




if __name__ == '__main__':
    
    df=pd.read_csv("Complete_DataSet_uncleaned.csv")
    headlines=df["headline"].tolist()
    articles=df["body"].tolist()
    label=df["fakeness"].tolist()
    res1=constructMat(headlines,label)
    
    my_df=pd.DataFrame(res1)
    
    
    my_df.to_csv('extracted1.csv',index=False,header=('NumOfWords', 'NumOfChar', 'NumOfPunc','NumOfContPunc','NumOfContUpperCase','Bi-freq','un-freq','tri-freq','Label'))
