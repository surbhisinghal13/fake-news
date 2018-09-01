import itertools
import numpy as np
import re
import os
import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

df=pd.read_csv("Complete_DataSet.csv")
MAX_NB_WORDS=20000
MAX_SEQUENCE_LENGTH=100
VALIDATION_SPLIT=0.2
EMBEDDING_DIM=100
def load_data_and_labels():
    
    headlines=df["headline"].tolist()
    print(headlines)
    articles=df["body"].tolist()
    y_label=df["fakeness"].tolist()
    
    return headlines,articles,y_label

texts,data,labels=load_data_and_labels()


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels)) # one hot encode for labels 

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

GLOVE_DIR="/home/surbhi/Desktop/minor1/minor2/glove.6B"

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
     
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding
from keras.layers import Dense,Input,Flatten,Conv1D,MaxPooling1D,Dropout
from keras.models import Model

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(16, 3, activation='relu')(embedded_sequences) #parameters- filter,kernel size
x = MaxPooling1D(2)(x)
x = Conv1D(32, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(64, 7, activation='relu')(x)
x = MaxPooling1D(2)(x)  # global max pooling
x = Flatten()(x)
x=Dropout(0.3)(x)
x = Dense(150, activation='relu')(x) # parameter 1: units : dimentionality of output space
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


model.fit(x_train, y_train, epochs=5, batch_size=128,shuffle=False)
score,acc = model.evaluate(np.array(x_val), np.array(y_val), verbose=0)
print("Accuracy= ",acc)




