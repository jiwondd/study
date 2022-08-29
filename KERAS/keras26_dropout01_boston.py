import numpy as np
import pandas as pd
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

#1. 데이터

datasets=load_boston()
x=datasets.data
y=datasets['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
model=Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.summary

'''
input1=Input(shape=(13,))
dense1=Dense(50)(input1)
dense2=Dense(100,activation='relu')(dense1)
drop1=Dropout(0.2)(dense2)
dense3=Dense(80,activation='relu')(drop1)
dense4=Dense(50,activation='relu')(dense3)
dense5=Dense(30,activation='relu')(dense4)
output1=Dense(1)(dense5)
model=Model(inputs=input1,outputs=output1)
'''

#3.컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date) #2022-07-07 17:50:42.752072
date=date.strftime('%m%d_%H%M')
print(date) #0707_1750

filepath='./_k24/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','boston',filename]))

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
               epochs=100, batch_size=100, verbose=1)


#4. 평가, 예측
print('==============1. 기본 출력값=====================')
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


# loss: 19.22631072998047
# r2스코어: 0.749943000884974

# loss: 18.06477928161621
# r2스코어: 0.7650499117982772 <-dropout 적용
