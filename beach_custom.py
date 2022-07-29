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
customer = pd.read_csv(path + '해운대 입장객수.csv')

# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

customer['방문일'] = pd.to_datetime(customer['방문일'], infer_datetime_format=True)
customer['방문객수(명)'] = customer['방문객수(명)'].str.replace(",","")
customer.iloc[:,1:] = customer.iloc[:,1:].apply(pd.to_numeric)

# merge data : 방문객수 예측을 위한 데이터
costom_data = pd.merge(weather, customer, left_on='날짜', right_on="방문일", how='inner')
costom_data = costom_data[['강수_관측값', "기온", "습도", "체감온도", "평균수온", "평균풍속", "평균기압", "평균최대파고", "평균파주기", "방문객수(명)"]]
print(costom_data.isnull().sum())


# weather data : 날씨 예측을 위한 데이터
costom_data.loc[costom_data['평균수온'] != costom_data['평균수온'], '평균수온'] = costom_data['평균수온'].mean()
costom_data.loc[costom_data['평균풍속'] != costom_data['평균풍속'], '평균풍속'] = costom_data['평균풍속'].mean()
costom_data.loc[costom_data['평균기압'] != costom_data['평균기압'], '평균기압'] = costom_data['평균기압'].mean()
costom_data.loc[costom_data['평균최대파고'] != costom_data['평균최대파고'], '평균최대파고'] = costom_data['평균최대파고'].mean()
costom_data.loc[costom_data['평균파주기'] != costom_data['평균파주기'], '평균파주기'] = costom_data['평균파주기'].mean()

train_x, test_x, cos_train_y, cos_test_y = train_test_split(costom_data.iloc[:, :-1], costom_data['방문객수(명)'], test_size=0.2)

costom_scaler = MinMaxScaler()
train_x_scaled = costom_scaler.fit_transform(train_x)
test_x_scaled = costom_scaler.transform(test_x)

print(train_x_scaled.shape,test_x_scaled.shape) #(265, 9) (67, 9)

cos_train=train_x_scaled.reshape(265, 3, 3)
cos_test=test_x_scaled.reshape(67, 3, 3)

# 2. 모델구성
cos_model=Sequential()
cos_model.add(LSTM(units=128,input_shape=(3,3)))
cos_model.add(Dense(256, activation='relu'))
cos_model.add(Dense(128, activation='relu'))
cos_model.add(Dense(64, activation='relu'))
cos_model.add(Dense(1))
cos_model.summary()


# 3. 컴파일 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
cos_model.compile(loss='mse',optimizer='adam')
hist=cos_model.fit(cos_train,cos_train_y,epochs=10,batch_size=16,validation_split=0.1, callbacks=[earlyStopping])
cos_model.save("./beach/costom_save_model.h5")


# 4. 평가, 예측
loss=cos_model.evaluate(cos_test,cos_test_y)
pred_y=cos_model.predict(cos_test)

print('loss: ',loss)
print('예상 입장객 수: ', pred_y[-1:])

# loss:  3082582272.0
# 예상 입장객 수:  [[85440.05]]
