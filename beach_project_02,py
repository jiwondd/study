import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
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
x1 = pd.read_csv(path + '해운대_날씨.csv')
x2 = pd.read_csv(path + '해운대_일일방문객.csv')

# x1['날짜'] = pd.to_datetime(x1['날짜'], infer_datetime_format=True)
# weather_set=x1(pd.to_numeric)

# x2['방문일'] = pd.to_datetime(x2['방문일'], infer_datetime_format=True)
# weather_set=x2(pd.to_numeric)

print(x1.shape,x2.shape) #(422, 9) (184, 2)
# print(x1.isnull().sum(),x2.isnull().sum())
x1.loc[x1['수온'] != x1['수온'], '수온'] = x1['수온'].mean()
x1.loc[x1['평균풍속'] != x1['평균풍속'], '평균풍속'] = x1['평균풍속'].mean()
x1.loc[x1['평균기압'] != x1['평균기압'], '평균기압'] = x1['평균기압'].mean()
x1.loc[x1['최대파고'] != x1['최대파고'], '최대파고'] = x1['최대파고'].mean()
print(x1.isnull().sum(),x2.isnull().sum())

x1['날짜'] = pd.to_datetime(x1['날짜'], infer_datetime_format=True)
weather_set=x1.apply(pd.to_numeric)

print(x1.shape) #(422, 9)
print(x2.shape) #(184, 2)


