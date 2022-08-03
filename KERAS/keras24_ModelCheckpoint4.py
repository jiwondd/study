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
model.add(Dense(64,input_dim=13))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))


#3.컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date)
date=date.strftime('%m%d_%H%M')
print(date)

filepath='./_ModelCheckPoint/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'


earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_',filename])
                    )
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
               epochs=100, batch_size=100, verbose=1)


#4. 평가, 예측
print('==============1. 기본 출력값=====================')
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)

# print('==============2. load_model 출력값=====================')
# model2=load_model('./_save/keras24_3_save_model.h5')
# loss2=model2.evaluate(x_test,y_test)
# print('loss2:',loss)
# y_predict2=model2.predict(x_test)
# r2=r2_score(y_test,y_predict2)
# print('r2스코어:',r2)

# print('==============3. ModelChekpoint 출력값=====================')
# model3=load_model('./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5')
# loss3=model3.evaluate(x_test,y_test)
# print('loss3:',loss)
# y_predict3=model3.predict(x_test)
# r2=r2_score(y_test,y_predict3)
# print('r2스코어:',r2)


