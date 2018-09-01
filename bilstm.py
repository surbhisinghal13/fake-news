import numpy as np
import re
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional
from keras.layers import Dense, Dropout, Activation,Embedding,SpatialDropout1D
import pandas as pd
from sklearn.model_selection import train_test_split

def transform_titles(text):
        
    data = list()
    for one_news in text:
        single = nltk.word_tokenize(clean_sentence(one_news))
        mapping = list()
        for one_keyword in single:
            mapping.append(one_hot(one_keyword, 7000)[0])
        data.append(mapping)
        # print(data)
    return data

def clean_sentence(s):
    c = s.lower().strip()
    return re.sub('[^a-z ]', '', c)

def save_model(text,label,unit_size = 10):
    batch_size = 10
        
    x_text = transform_titles(text)
        
    batch_size = 32
    vocabulary_size=20000
    tokenizer=Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    data=pad_sequences(sequences,maxlen=50)
        
       
    return data,label

def lstm_model(x_train,y_train):
    batch_size=32
    model = Sequential()
        
    model.add(Embedding(50000, 128))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(np.array(x_train), np.array(y_train),
              epochs=3,
              batch_size=batch_size,
              shuffle=False)
    return model
   

if __name__ == "__main__":
    
    df=pd.read_csv("Complete_DataSet.csv")
    headlines=df["headline"].tolist()
    articles=df["body"].tolist()
    label=df["fakeness"].tolist()
    x=int(input("enter 0: for headlines or 1: for content"))
    if x==0:
        data=headlines
    else:
        data=articles
    x,y=save_model(data,label)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=3)
 
    model=lstm_model(x_train,y_train)
    score,acc = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
    print("accuracy= ",acc)
    

    
    
    

