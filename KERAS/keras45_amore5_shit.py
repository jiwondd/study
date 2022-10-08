import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Input, concatenate, Conv1D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
import math
import time

path='./_data/test_amore_0718/'
x1_data=pd.read_csv(path+'삼성전자220718.csv',encoding='cp949')
x2_data=pd.read_csv(path+'아모레220718.csv',encoding='cp949')

# print(x1_data)
# print(x1_data.shape) (3040, 17)
# print(x2_data)
# print(x2_data.shape) (3180, 17)

x1_data=x1_data.sort_values(by=['일자'], ascending=[True])
x2_data=x2_data.sort_values(by=['일자'], ascending=[True])

x1_data = x1_data.drop(['Unnamed: 6','전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)
x2_data = x2_data.drop(['Unnamed: 6','전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

x1_data = x1_data.fillna(0)
x2_data = x2_data.fillna(0)

x1_data = x1_data.loc[x1_data['일자']>="2018/05/04"] 
x2_data = x2_data.loc[x2_data['일자']>="2018/05/04"] 
# print(x1_data.shape, x2_data.shape) #(1035, 9) (1035, 9)
# x1_data=x1_data.drop(labels=range(1037,3040),axis=0)
# x2_data=x2_data.drop(labels=range(1037,3180),axis=0)
# print(x1_data.shape, x2_data.shape) #(1037, 9) (1037, 9)

x1_data = x1_data.sort_values(by=['일자'], ascending=True) 
x2_data = x2_data.sort_values(by=['일자'], ascending=True)

feature_cols = ['시가', '고가', '저가', '거래량', '기관', '외국계', '종가']
label_cols = ['시가']

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

SIZE = 20
x1 = split_x(x1_data[feature_cols], SIZE)
x2 = split_x(x2_data[feature_cols], SIZE)
y = split_x(x2_data[label_cols], SIZE)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.3, shuffle=True, random_state=42)
print(x1_train.shape,x1_test.shape) #(711, 20, 7) (305, 20, 7)
print(x2_train.shape,x2_test.shape) #(711, 20, 7) (305, 20, 7)
print(y_train.shape,y_test.shape) #(711, 20, 7) (305, 20, 7)

# scaler = MinMaxScaler()
# x1_train = x1_train.reshape(711*20,7)
# x1_train = scaler.fit_transform(x1_train)
# x1_test = x1_test.reshape(305*20,7)
# x1_test = scaler.transform(x1_test)

# x2_train = x2_train.reshape(711*20,7)
# x2_train = scaler.fit_transform(x2_train)
# x2_test = x2_test.reshape(305*20,7)
# x2_test = scaler.transform(x2_test)

x1_train = x1_train.reshape(711, 20, 7)
x1_test = x1_test.reshape(305, 20, 7)
x2_train = x2_train.reshape(711, 20, 7)
x2_test = x2_test.reshape(305, 20, 7)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(20, 7))
dense1 = Conv1D(64, 2, activation='relu')(input1)
dense2 = LSTM(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
output1 = Dense(32, activation='relu')(dense3)

# 2-2. 모델2
input2 = Input(shape=(20, 7))
dense11 = Conv1D(64, 2, activation='relu')(input2)
dense12 = LSTM(128, activation='swish')(dense11)
dense13 = Dense(64, activation='relu')(dense12)
dense14 = Dense(32, activation='relu')(dense13)
output2 = Dense(16, activation='relu')(dense14)

merge1 = concatenate([output1, output2])
merge2 = Dense(100, activation='relu')(merge1)
merge3 = Dense(100)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=64, callbacks=[Es], validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:])

# Node: 'model/Cast'
# Cast string to float is not supported
#          [[{{node model/Cast}}]] [Op:__inference_train_function_7890] 
