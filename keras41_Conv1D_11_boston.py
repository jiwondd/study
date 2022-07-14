import numpy as np
import pandas as pd
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten,MaxPooling2D,Dropout, LSTM
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

print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

print(x_train.shape,x_test.shape) #(404, 13) (102, 13)

scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train=x_train.reshape(404,13,1)
x_test=x_test.reshape(102,13,1)

#2. 모델구성
model=Sequential()
model.add(Conv1D(10,2,input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.summary

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('boston_끝났어요')


# loss: 17.8665771484375
# r2스코어: 0.7676276855828383

# loss: 19.355871200561523
# r2스코어: 0.7482579665877793 <-LSTM으로 바꾼거 