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

path='./_beach/'
weather = pd.read_csv(path + '해운대 날씨.csv')

# print(weather.head(5))
print(weather.shape) #(1618, 10)

# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

# print(weather.isnull().sum())
weather.loc[weather['평균풍속'] != weather['평균풍속'], '평균풍속'] = weather['평균풍속'].mean()
weather.loc[weather['평균기압'] != weather['평균기압'], '평균기압'] = weather['평균기압'].mean()
weather.loc[weather['평균수온'] != weather['평균수온'], '평균수온'] = weather['평균수온'].mean()
weather.loc[weather['평균최대파고'] != weather['평균최대파고'], '평균최대파고'] = weather['평균최대파고'].mean()
weather.loc[weather['평균파주기'] != weather['평균파주기'], '평균파주기'] = weather['평균파주기'].mean()
# print(weather.isnull().sum())

# print(type(weather))

print(type(weather))


'''
# train/test split
train_x, test_x, train_y, test_y = train_test_split(weather.iloc[:, :-1], weather['기온'], test_size=0.2)

# print(train_x.shape) #(1294, 9)
# print(test_x.shape) #(324, 9)

# minmax scaler
x_mm_scaler = MinMaxScaler()
train_x_scaled = x_mm_scaler.fit_transform(train_x)
test_x_scaled = x_mm_scaler.transform(test_x)

print(train_x.shape) #(1294, 9)
print(test_x.shape) #(324, 9)

# TypeError: The DTypes <class 'numpy.dtype[float64]'> and <class 'numpy.dtype[datetime64]'> do not have a common DType. 
# For example they cannot be stored in a single array unless the dtype is `object`.
'''