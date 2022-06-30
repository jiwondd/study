from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
datasets=load_boston()
x=datasets.data
y=datasets.target

#1. 데이터
# print(x)
# print(y)
# print(x.shape, y.shape) #(506, 13) (506,)

# print(datasets.feature_names)
# print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.9,shuffle=True, random_state=31)


# [실습] 아래를 완성할 것
# #1. train 0.7
# #2. R2 0.8 이상

#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=13))
model.add(Dense(40))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train, epochs=300, batch_size=10,validation_split=0.15)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


'''
loss: 17.545747756958008
r2스코어: 0.7876257019581974 <-validation 없었을때
********************************

loss: 26.963623046875
r2스코어: 0.680930610231024 <-validation 넣었을때

loss: 31.915023803710938
r2스코어: 0.5836996498410041

loss: 37.301185607910156
r2스코어: 0.5586028063696656

loss: 44.82423782348633
r2스코어: 0.41531149575503745 train 0.8 / val 0.5

loss: 32.96104431152344
r2스코어: 0.5700552810438276 train 0.8 / val 0.15

loss: 16.75043296813965
r2스코어: 0.7422860063867474 train 0.9 / val 0.15




'''