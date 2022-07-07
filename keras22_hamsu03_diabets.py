from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=72)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


#2. 모델구성
# model=Sequential()
# model.add(Dense(50,activation='relu',input_dim=10))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(50,activation='linear'))
# model.add(Dense(50,activation='linear'))
# model.add(Dense(1))

input1=Input(shape=(10,))
dense1=Dense(50)(input1)
dense2=Dense(100,activation='elu')(dense1)
dense3=Dense(100,activation='relu')(dense2)
dense4=Dense(50,activation='relu')(dense3)
dense5=Dense(80,activation='linear')(dense4)
output1=Dense(1)(dense5)
model=Model(inputs=input1,outputs=output1)

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=3000, batch_size=100, verbose=1)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


'''
loss: 1080.822998046875
r2스코어: 0.746747006658822 <-기존

loss: 2319.88671875
r2스코어: 0.6486733527571984 <-MinMax

loss: 2764.81640625
r2스코어: 0.5812925903625359 <-Standard

loss: 2334.639404296875
r2스코어: 0.6464391759478613 <-MaxAbsScaler

loss: 2928.393310546875
r2스코어: 0.556520325077233<-RobustScaler

loss: 2480.673583984375
r2스코어: 0.6243235503060469 <-함수


'''