#summary, time을 알아보자!
# 에포는 10개

import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import time

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target
print(x.shape,y.shape) #(581012, 54) (581012,)
# print(np.unique(y)) #[1 2 3 4 5 6 7]
print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
                                    # dtype=int64)) 
# y=to_categorical(y)
# print(y) 
# print(x.shape,y.shape) #(581012, 54) (581012, 8) -> 투카테고리얼을 쓰면 

y=pd.get_dummies(y)
print(x.shape,y.shape) #(581012, 54) (581012, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

#2. 모델구성
model=Sequential()
model.add(Dense(120,input_dim=54))
model.add(Dense(200,activation='relu'))
model.add(Dense(250,activation='relu'))
model.add(Dense(180,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(140,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='linear'))
model.add(Dense(7,activation='softmax'))

#3. 컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto',verbose=1,restore_best_weights=True) #
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
start_time=time.time()
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=10, batch_size=100, verbose=1)
end_time=time.time()-start_time

#4. 평가, 예측

result=model.evaluate(x_test,y_test)
print('loss:',result[0])
print('accuracy:',result[1])
print("===================================")
y_predict=model.predict(x_test)
y_predict=tf.argmax(y_test,axis=1)
y_test=tf.argmax(y_test,axis=1)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
print("걸린시간:",end_time)
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 120)               6600
_________________________________________________________________
dense_1 (Dense)              (None, 200)               24200
_________________________________________________________________
dense_2 (Dense)              (None, 250)               50250
_________________________________________________________________
dense_3 (Dense)              (None, 180)               45180
_________________________________________________________________
dense_4 (Dense)              (None, 150)               27150
_________________________________________________________________
dense_5 (Dense)              (None, 140)               21140
_________________________________________________________________
dense_6 (Dense)              (None, 120)               16920
_________________________________________________________________
dense_7 (Dense)              (None, 100)               12100
_________________________________________________________________
dense_8 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_9 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_10 (Dense)             (None, 7)                 707
=================================================================
Total params: 224,447
Trainable params: 224,447
Non-trainable params: 0

'''