from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target
# print(x.shape,y.shape)
# print(np.unique(y,return_counts=True)) 
# y=to_categorical(y)
# print(y) 
# print(x.shape,y.shape) #(581012, 54) (581012, 8) -> 투카테고리얼을 쓰면 

y=pd.get_dummies(y)
# print(x.shape,y.shape) #(581012, 54) (581012, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(120,input_dim=54))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(160,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(80,activation='linear'))
# model.add(Dense(7,activation='softmax'))

input1=Input(shape=(54,))
dense1=Dense(120)(input1)
dense2=Dense(200,activation='relu')(dense1)
dense3=Dense(160,activation='relu')(dense2)
dense4=Dense(100,activation='relu')(dense3)
dense5=Dense(80,activation='linear')(dense4)
output1=Dense(7,activation='softmax')(dense5)
model=Model(inputs=input1,outputs=output1)

#3. 컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto',verbose=1,restore_best_weights=True) #
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=10, batch_size=100, verbose=1)


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


'''
loss: 0.4436834752559662
accuracy: 0.812620997428894
===================================
acc score : 1.0 <-기존

loss: 0.24028544127941132
accuracy: 0.9033674001693726
===================================
acc score : 1.0 <-MinMax

loss: 0.24028544127941132
accuracy: 0.9033674001693726
===================================
acc score : 1.0 <-standard

loss: 0.24165020883083344
accuracy: 0.9014397263526917
===================================        
acc score : 1.0   <-MaxAbsScaler
   
loss: 0.19975309073925018
accuracy: 0.920501172542572
===================================
acc score : 1.0<-RobustScaler

loss: 0.19935636222362518
accuracy: 0.9197180867195129
===================================
acc score : 1.0 <-함수

'''
