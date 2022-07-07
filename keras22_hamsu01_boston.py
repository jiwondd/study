from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np


datasets=load_boston()
x=datasets.data
y=datasets['target']

# print(np.min(x)) #0.0 (최소값 뽑는 함수)
# print(np.max(x)) #711.0 (최대값 뽑는 함수)
# x=(x-np.min(x))/(np.max(x)-np.min(x))
# print(x[:10])          <-처음 예시로 했던 부분임 / 트레인테스트 전에 하면 안됨
x_train, x_test, y_train,y_test=train_test_split(
    x,y, train_size=0.7, random_state=777)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# print(np.min(x_train)) #0.0
# print(np.max(x_train)) #1.0000000000000002
# print(np.min(x_test)) #0.0
# print(np.max(x_test)) #666.0

#2. 모델구성
# model=Sequential()
# model.add(Dense(50,activation='relu',input_dim=13))
# model.add(Dense(100,activation='relu')) 
# model.add(Dense(80,activation='relu'))
# model.add(Dense(50,activation='elu'))
# model.add(Dense(30,activation='linear'))
# model.add(Dense(1))

input1=Input(shape=(13,))
dense1=Dense(50)(input1)
dense2=Dense(100,activation='relu')(dense1)
dense3=Dense(80,activation='relu')(dense2)
dense4=Dense(50,activation='relu')(dense3)
dense5=Dense(30,activation='relu')(dense4)
output1=Dense(1)(dense5)
model=Model(inputs=input1,outputs=output1)

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)

'''
loss:  16.85349464416504
r2스코어: 0.770180980297865 <-기존

loss: 13.69657039642334
r2스코어: 0.8379239911526317 <-MinMax

loss: 14.206235885620117
r2스코어: 0.8318929467685272 <-Standard

loss: 13.248407363891602
r2스코어: 0.8432272402637725 <-MaxAbs

loss: 15.63769817352295
r2스코어: 0.8149539754790187 <-Robust

loss: 24.359905242919922
r2스코어: 0.7117412567109576 <-함수


'''




