import numpy as np
import pandas as pd
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

#1. 데이터

datasets=load_boston()
x,y=datasets.data, datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(64,input_dim=13))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1))


# #3.컴파일, 훈련
# earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
#                             verbose=1,restore_best_weights=True)
# model.compile(loss='mse',optimizer="adam")
# mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
#                     save_best_only=True, 
#                     filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')
# hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
#                epochs=100, batch_size=100, verbose=1)


model=load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)





'''
loss: 19.987375259399414
r2스코어: 0.7400446267925227 <-기존

loss: 527.5485229492188
r2스코어: -5.861284337153851 <-훈련시키기 전의 랜덤한 웨이트로 돌린 값

loss: 527.5485229492188
r2스코어: -5.861284337153851 <- 훈련시켜서 나온 웨이트로 돌린 값

loss: 17.624465942382812
r2스코어: 0.7707765723109106 

loss: 17.853858947753906
r2스코어: 0.7677931191164589 <- MOYA...왜 달라...

-------------------------------------

loss: 19.444578170776367
r2스코어: 0.747104232972083

loss: 19.444578170776367
r2스코어: 0.747104232972083 <- 다시 하니까 똑같이 나옴

'''
