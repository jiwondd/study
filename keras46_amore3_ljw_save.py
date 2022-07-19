import numpy as np
import pandas as pd
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
x1_data = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
x2_data = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

x1_data = x1_data.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)
x2_data = x2_data.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

x1_data = x1_data.fillna(0)
x2_data = x2_data.fillna(0)

x1_data = x1_data.loc[x1_data['일자']>="2018/05/04"] 
x2_data = x2_data.loc[x2_data['일자']>="2018/05/04"] 
print(x1_data.shape, x2_data.shape) # (1035, 11) (1035, 11)

x1_data = x1_data.sort_values(by=['일자'], axis=0, ascending=True)
x2_data = x2_data.sort_values(by=['일자'], axis=0, ascending=True)

feature_cols = ['시가', '고가', '저가', '거래량', '기관', '외국계', '종가']
label_cols = ['종가']

def split_xy3(dataset_x, dataset_y, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset_x)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset_x):
            break
        tmp_x = dataset_x[i:x_end_number]
        tmp_y = dataset_y[x_end_number: y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

SIZE = 3
COLSIZE = 3
x1, y1 = split_xy3(x1_data[feature_cols], x1_data[label_cols], SIZE, COLSIZE)
x2, y2 = split_xy3(x2_data[feature_cols], x2_data[label_cols], SIZE, COLSIZE)
# print(x1.shape, y1.shape) #(1030, 3, 7) (1030, 3, 1)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, shuffle=True)

scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) # (824, 3, 7) (206, 3, 7)
print(x2_train.shape, x2_test.shape) # (824, 3, 7) (206, 3, 7)
print(y_train.shape, y_test.shape) # (824, 3, 1) (206, 3, 1)

x1_train = x1_train.reshape(824*3,7)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(206*3,7)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(824*3,7)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(206*3,7)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(824, 3, 7)
x1_test = x1_test.reshape(206, 3, 7)
x2_train = x2_train.reshape(824, 3, 7)
x2_test = x2_test.reshape(206, 3, 7)


# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(3, 7))
dense1 = Conv1D(64, 2, activation='relu')(input1)
dense2 = LSTM(128, activation='relu',return_sequences=True)(dense1)
dense3 = LSTM(256, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(64)(dense4)
output1 = Dense(32)(dense5)

# 2-2. 모델2
input2 = Input(shape=(3, 7))
densea = Conv1D(64, 2, activation='relu')(input2)
denseb = LSTM(128, activation='relu',return_sequences=True)(densea)
densec = LSTM(256, activation='relu')(denseb)
densed = Dense(128, activation='relu')(densec)
densee = Dense(64)(densed)
output2 = Dense(32)(densee)

merge1 = concatenate([output1, output2])
merge2 = Dense(128, activation='relu')(merge1)
merge3 = Dense(64, activation='relu')(merge2)
merge4 = Dense(32, activation='relu')(merge3)
merge5 = Dense(16)(merge4)
last_output = Dense(1)(merge5)
model = Model(inputs=[input1, input2], outputs=[last_output])
model.summary

# 3. 컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date)
date=date.strftime('%m%d_%H%M')
print(date)

model.compile(loss='mse', optimizer='adam')
filepath='./_k24/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','amore_jongga',filename]))
model.fit([x1_train, x2_train], y_train, epochs=2000, batch_size=1024, callbacks=[Es,mcp], validation_split=0.2)
model.save("./_test/keras46_amore3_lij_save.h5")

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:])

# loss:  217945552.0
# prdict:  [[135984.67]]