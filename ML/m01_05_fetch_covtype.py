# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.svm import LinearSVC

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=MinMaxScaler()
scaler=StandardScaler()
scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
model=LinearSVC()

#3. 컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
print('결과: ',result)
y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc) # =결과 result


'''

loss: 0.19975309073925018
accuracy: 0.920501172542572
===================================
acc score : 1.0<-RobustScaler

결과:  0.711212275070351
acc score : 0.711212275070351 <-LinearSVC


'''