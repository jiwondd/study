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

# 필요없는 (잘모르는) 컬럼 제거하기 
x1_data=x1_data.drop(['전일비','등락률','신용비','개인','외인비','기관',
              '외인(수량)','외국계','프로그램'],axis=1)
x2_data=x2_data.drop(['전일비','등락률','신용비','개인','외인비','기관',
              '외인(수량)','외국계','프로그램'],axis=1)
x1_data=x1_data.drop(labels=range(1037,3040),axis=0)
x2_data=x2_data.drop(labels=range(1037,3180),axis=0)
# print(x1_data.shape) (1037, 8)
# print(x2_data.shape) (1037, 8)


# 날짜 데이터 데이트타임으로 바꾸기 
x1_data['일자']=pd.to_datetime(x1_data['일자'])
x1_data['year']=x1_data['일자'].dt.year
x1_data['month']=x1_data['일자'].dt.month
x1_data['day']=x1_data['일자'].dt.day

x2_data['일자']=pd.to_datetime(x2_data['일자'])
x2_data['year']=x2_data['일자'].dt.year
x2_data['month']=x2_data['일자'].dt.month
x2_data['day']=x2_data['일자'].dt.day

cols = ['year','month','day']
for col in cols:
    le = LabelEncoder()
    x1_data[col]=le.fit_transform(x1_data[col])
    x2_data[col]=le.fit_transform(x2_data[col])

x1=np.array(x1_data)
x2=np.array(x2_data)

# print(x1_data.shape) #(1037, 11)
# print(x2_data.shape) #(1037, 11)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column 

        if y_end_number > len(dataset): 
            break
        tmp_x = dataset[i:x_end_number, :]  
        tmp_y = dataset[x_end_number:y_end_number, 3]  
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy3(x1, 5, 1) 
x2, y = split_xy3(x2, 5, 1) 
# print(x2[0,:], "\n", y[0])
# print(x2.shape) #(1032, 5, 11)
# print(y.shape) #(1032, 1)


x1_train, x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1,x2,y,train_size=0.7, shuffle=True, random_state=777)

print(x1_train.shape,x1_test.shape) #(722, 5, 11) (310, 5, 11)
print(x2_train.shape,x2_test.shape) #(722, 5, 11) (310, 5, 11)
print(y_train.shape,y_test.shape) #(722, 1) (310, 1)

# 2. 모델구성

# 2-1 모델1
input1=Input(shape=(5,11))
dense1=LSTM(64)(input1)
dense2=Dense(128)(dense1)
dense3=Dense(64)(dense2)
output1=Dense(1)(dense3)

# 2-2 모델2
input2=Input(shape=(5,11))
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

# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type Timestamp).
