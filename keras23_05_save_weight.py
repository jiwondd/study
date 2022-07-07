from tabnanny import verbose
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터

datasets=load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()

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
model.summary()
# model.save("./_save/keras23_1_save_model.h5")
model.save_weights("./_save/keras23_5_save_weight1.h5") #얘는 핏 훈련을 안해서 랜던 웨이트가 저장되겠죠?
# model=load_model("./_save/keras23_3_save_model.h5")

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)
# model.save("./_save/keras23_3_save_model.h5")
model.save_weights("./_save/keras23_5_save_weight2.h5")
# model=load_model("./_save/keras23_3_save_model.h5")

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


'''
loss: 16.297264099121094
r2스코어: 0.788038151691532 <- 기존값

loss: 16.297264099121094
r2스코어: 0.788038151691532 <-keras23_3_save.model에서 fit 결과까지 저장하고 불러온 값 (위랑 똑같)

loss: 16.585771560668945
r2스코어: 0.7842858170915608 <- 모델만 불러오고 훈련은 다시 한 결과 

'''
