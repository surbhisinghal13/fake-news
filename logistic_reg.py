import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def accuracy(predict, test_label):
    acc = 0
   # print(test_label)
    for i in range(len(predict)):
        if predict[i] == test_label[i]:
            acc += 1
    print("acc = ",acc,"/",len(predict))
    return acc*1.0/len(predict)


df1=pd.read_csv("extracted1.csv")
df_x = df1[df1.columns[0:8]].values
df_y = df1[df1.columns[8]]

  
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20,random_state=24)
y_train=pd.factorize(y_train)[0]
y_test=pd.factorize(y_test)[0]   
clf=LogisticRegression()
clf=clf.fit(x_train,y_train)
pred=clf.predict(x_test)
acc=accuracy(pred,y_test)
    
print("accuracy of LR= ",acc)
