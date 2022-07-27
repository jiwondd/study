import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time
from sklearn.model_selection import train_test_split
import inspect, os

path = './_beach/'
weather_set = pd.read_csv(path + '해운대_날씨.csv')


# print(weather_set.head) # [422 rows x 9 columns]

# weather_set = weather_set.fillna(weather_set.mean())

# 날짜 데이터 데이터 프레임으로 변환하기
weather_set['날짜'] = pd.to_datetime(weather_set['날짜'], infer_datetime_format=True)
weather_set=weather_set.apply(pd.to_numeric)

print(weather_set.shape) #(422, 9)

feature_cols = ['날짜','강수_관측값','기온','습도','체감온도','수온','평균풍속','평균기압','최대파고']
label_cols = ['강수_관측값','기온']

def split_xy2(weather_set,time_steps,y_column):
    x,y=list(), list()
    for i in range(len(weather_set)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_column
        if y_end_number > len(weather_set):
            break
        tmp_x=weather_set[i:x_end_number]
        tmp_y=weather_set[x_end_number:y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x),np.array(y)

time_steps=9
y_column=2
x,y=split_xy2(weather_set,time_steps,y_column)
print(x.shape) #(412, 9, 9)
print(y.shape) #(412, 2, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

print(x_train.shape,y_train.shape) #(329, 9, 9) (329, 3, 9)
print(x_test.shape,y_test.shape) #(83, 9, 9) (83, 3, 9)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()

x_train = x_train.reshape(329*9,9)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(83*9,9)
x_test = scaler.transform(x_test)

x_train=x_train.reshape(329,9,9)
x_test=x_test.reshape(83,9,9)

print(x_train.shape,y_train.shape) #(329, 9, 9) (329, 2, 9)
print(x_test.shape,y_test.shape) #(83, 9, 9) (83, 2, 9)
 
# print(x_train.shape,y_train.shape) #
# print(x_test.shape,y_test.shape) #


# #2.모델 구성
# model = Sequential()
# model.add(LSTM(units=32,return_sequences=True,input_shape=(9,9)))
# model.add(LSTM(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='linear'))
# model.add(Dense(32,activation='linear'))
# model.add(Dense(1))
# model.summary()

# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train,y_train,epochs=10)

input1=Input(shaape=(9,9))
LSTM1=LSTM(32, activation='relu',return_sequences=True)(input1)
LSTM2=LSTM(64, activation='relu')(LSTM1)
dense1= Dense(128, activation='relu')(LSTM2)
dense2= Dense(256, activation='relu')(dense1)
dense3= Dense(128, activation='relu')(dense2)
dense4= Dense(64, activation='relu')(dense3)
dense5= Dense(32, activation='relu')(dense4)
dense6= Dense(16, activation='relu')(dense5)
output1=Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)
model.summary()



