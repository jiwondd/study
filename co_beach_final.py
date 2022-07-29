# -*- coding: utf-8 -*-
"""co_beach_final

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K0bKvRi-GXtFC0Va3fofeEpWutg434UU
"""

# google drive 연동
from google.colab import drive
drive.mount('/content/drive/')

import os, pandas as pd
os.chdir('/content/drive/MyDrive/jiwon')

weather = pd.read_csv('해운대 날씨.csv')
customer = pd.read_csv('해운대 입장객수2.csv')

weather.head(5)

customer.head(5)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
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


# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

customer['방문일'] = pd.to_datetime(customer['방문일'], infer_datetime_format=True)
customer['방문객수'] = customer['방문객수'].str.replace(",","")
customer.iloc[:,1:] = customer.iloc[:,1:].apply(pd.to_numeric)

weather.shape

customer.shape

# merge data : 방문객수 예측을 위한 데이터
total_data = pd.merge(weather, customer, left_on='날짜', right_on="방문일", how='inner')
total_data = total_data[['강수_관측값', "기온", "습도", "체감온도", "평균수온", "평균풍속", "평균기압", "평균최대파고", "평균파주기", "방문객수"]]

# weather data : 날씨 예측을 위한 데이터(na는 제거하는 방향으로 일단 진행)
total_data.loc[total_data['평균수온'] != total_data['평균수온'], '평균수온'] = total_data['평균수온'].mean()
total_data.loc[total_data['평균풍속'] != total_data['평균풍속'], '평균풍속'] = total_data['평균풍속'].mean()
total_data.loc[total_data['평균기압'] != total_data['평균기압'], '평균기압'] = total_data['평균기압'].mean()
total_data.loc[total_data['평균최대파고'] != total_data['평균최대파고'], '평균최대파고'] = total_data['평균최대파고'].mean()
total_data.loc[total_data['평균파주기'] != total_data['평균파주기'], '평균파주기'] = total_data['평균파주기'].mean()
total_data = total_data.fillna(0)

total_data.isnull().sum()

"""## 방문객 수 예측"""

# 방문객 수 예측을 위한 train/test split
train_x, test_x, train_y, test_y = train_test_split(total_data.iloc[:, :-1], total_data['방문객수'], test_size=0.2)

# 방문객 수 예측을 위한  minmax scaler
x_mm_scaler = MinMaxScaler()
train_x_scaled = x_mm_scaler.fit_transform(train_x)
test_x_scaled = x_mm_scaler.transform(test_x)

train_x_scaled.shape

test_x_scaled.shape

# 방문객 수 예측을 위한  model training
model=Sequential()
model.add(Dense(128,input_dim=9, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()

# 학습 진행
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer='adam')
hist=model.fit(train_x_scaled,train_y,epochs=3000,batch_size=128,validation_split=0.2, callbacks=[earlyStopping])

loss=model.evaluate(test_x_scaled,test_y)
pred_y=model.predict(test_x_scaled)

print('loss: ',loss)
print('예상 입장객 수: ', pred_y[-1:])

"""## 강수_관측값 예측"""

# 사용 데이터
weather.info()

# input data
weather_input = weather[['강수_관측값',"기온", "습도", "체감온도", "평균풍속","평균기압", "평균수온", "평균최대파고", "평균파주기"]]
weather_output = weather[['강수_관측값',"기온"]]

# 강수_관측값 예측을 위한 train/test split
train_x, test_x, train_y, test_y = train_test_split(weather_input.values, weather_output.values, test_size=0.2)

# 강수_관측값 예측을 위한  minmax scaler
x_mm_scaler = MinMaxScaler()
train_x_scaled = x_mm_scaler.fit_transform(train_x)
test_x_scaled = x_mm_scaler.transform(test_x)

train_x_scaled.shape

test_x_scaled.shape

# 강수_관측값 예측을 위한 data reshape
train_x_scaled = train_x_scaled.reshape(1294, 3, 3)
test_x_scaled = test_x_scaled.reshape(324, 3, 3)

# 강수_관측값 예측을 위한 lstm 학습
model=Sequential()
model.add(LSTM(units=128,input_shape=(3,3)))
model.add(Dense(64, activation='relu'))
model.add(Dense(2))
model.summary()

# 학습 진행
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
hist = model.fit(train_x_scaled, train_y, epochs=5, batch_size=16, 
                validation_split=0.2,
                callbacks = [earlyStopping],
                verbose=2)

loss=model.evaluate(test_x_scaled,test_y)
pred_y=model.predict(test_x_scaled)

print('loss: ',loss)
print('예상 강수_관측값,기온 : ', pred_y[-1:])

"""## 기온 예측"""

# 사용 데이터
weather.info()

# input data
weather_input = weather[["강수_관측값", "습도", "체감온도", "평균수온", "평균풍속", "평균기압", "평균최대파고", "평균파주기"]]
weather_output = weather[['기온']]

# 기온 예측을 위한 train/test split
train_x, test_x, train_y, test_y = train_test_split(weather_input.values, weather_output.values, test_size=0.2)

# 기온 예측을 위한  minmax scaler
x_mm_scaler = MinMaxScaler()
train_x_scaled = x_mm_scaler.fit_transform(train_x)
test_x_scaled = x_mm_scaler.transform(test_x)

# 기온 예측을 위한 data reshape
train_x_scaled = train_x_scaled.reshape(-1, 8, 1)
test_x_scaled = test_x_scaled.reshape(-1, 8, 1)

# 기온 예측을 위한 lstm 학습
model=Sequential()
model.add(LSTM(units=128,input_shape=(8,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()

# 학습 진행
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
hist = model.fit(train_x_scaled, train_y, epochs=500, batch_size=16, 
                validation_split=0.2,
                callbacks = [earlyStopping],
                verbose=2)

loss=model.evaluate(test_x_scaled,test_y)
pred_y=model.predict(test_x_scaled)

print('loss: ',loss)
print('예상 기온 예측값 : ', pred_y[-1:])

