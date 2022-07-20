import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
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

x1_data['일자'] = pd.to_datetime(x1_data['일자'])
# x1_data['연도']=x1_data['일자'].dt.year
x2_data['일자'] = pd.to_datetime(x2_data['일자'])
# x2_data['연도']=x2_data['일자'].dt.year

x1_data.insert(0,'연',x1_data['일자'].dt.year)
x1_data.insert(1,'월',x1_data['일자'].dt.month)
x1_data.insert(2,'일',x1_data['일자'].dt.day)

x2_data.insert(0,'연',x1_data['일자'].dt.year)
x2_data.insert(1,'월',x1_data['일자'].dt.month)
x2_data.insert(2,'일',x1_data['일자'].dt.day)
# print(x1_data.shape) (3040, 20)
# print(x2_data.shape) (3180, 20)

# 필요없는 (잘모르는) 컬럼 제거하기 
x1_data=x1_data.drop(['일자','전일비','Unnamed: 6','등락률','신용비','개인','외인비',
              '외인(수량)','외국계','프로그램'],axis=1)
x2_data=x2_data.drop(['일자','전일비','Unnamed: 6','등락률','신용비','개인','외인비',
              '외인(수량)','외국계','프로그램'],axis=1)
x1_data=x1_data.drop(labels=range(1037,3040),axis=0)
x2_data=x2_data.drop(labels=range(1037,3180),axis=0)
print(x1_data.shape) #(1037, 10)
print(x2_data.shape) #(1037, 10)


feature_cols = ['0','1','2','시가', '고가', '저가', '거래량', '기관', '금액(백만)']
label_cols = ['시가']
print(x1_data.shape) #(1037, 10)
print(x2_data.shape) #(1037, 10)
size = 2                                      
def split_x1(dataset, size):                   
    a1 = []                                  
    for i in range(len(dataset) - size + 1):   
        subset = dataset[i : (i + size)]      
        a1.append(subset)                     
    return np.array(a1)                      
x1 = split_x1(x1_data, size) 
x2 = split_x1(x2_data, size)     
y = split_x1(x2_data['시가'], size) 
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
    train_size=0.7)
# print(x1_train.shape,x1_test.shape) #(725, 2, 10) (311, 2, 10
# print(x2_train.shape,x2_test.shape) 
# print(y_train.shape,y_test.shape) #(725, 2) (311, 2)

scaler = MinMaxScaler()
# scaler = StandardScaler()  
# scaler = MaxAbsScaler()                                                                                  
# scaler = RobustScaler()
x1_train = scaler.fit_transform(x1_train) 
x2_train = scaler.fit_transform(x2_train) 
x1_test = scaler.transform(x1_test)
x2_test = scaler.transform(x2_test)

print(x1_train.shape,x1_test.shape) #(725, 2, 10) (311, 2, 10
print(x2_train.shape,x2_test.shape) 
print(y_train.shape,y_test.shape) #(725, 2) (311, 2)


# 2. 모델구성

# 2-1 모델1
input1=Input(shape=(5,10))
dense1=LSTM(64)(input1)
dense2=Dense(128)(dense1)
dense3=Dense(64)(dense2)
output1=Dense(1)(dense3)

# 2-2 모델2
input2=Input(shape=(5,10))
densea=LSTM(64)(input2)
denseb=Dense(128)(densea)
densec=Dense(64)(denseb)
output2=Dense(1)(densec)

merge1=Concatenate()([output1,output2])
merge2=Dense(64)(merge1)
merge3=Dense(64)(merge2)
last_out=Dense(1)(merge3)
model=Model(inputs=[input1,input2],outputs=last_out)
model.summary()

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train, x2_train], y_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=10, 
          callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)

print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x1_test, x2_test])

#ValueError: could not convert string to float: '47,400'
