import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
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
customer = pd.read_csv(path + '해운대 입장객수2.csv')

print(weather.shape,customer.shape) #(1618, 10) (1465, 2)

# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

customer['방문일'] = pd.to_datetime(customer['방문일'], infer_datetime_format=True)
customer['방문객수'] = customer['방문객수'].str.replace(",","")
customer.iloc[:,1:] = customer.iloc[:,1:].apply(pd.to_numeric)

total_data = pd.merge(weather, customer, left_on='날짜', right_on="방문일", how='inner')
total_data = total_data[['강수_관측값', "기온", "습도", "체감온도", "평균수온", "평균풍속", "평균기압", "평균최대파고", "평균파주기", "방문객수"]]

print(weather.shape,customer.shape)

# input, output 지정
input_set = weather[["강수_관측값", "기온", "습도", "체감온도", "평균풍속", "평균기압", "평균수온","평균최대파고", "평균파주기"]]
output_set = weather[['강수_관측값','기온']]

# costom_input = weather[["강수_관측값", "기온", "습도", "체감온도", "평균풍속", "평균기압", "평균수온","평균최대파고", "평균파주기",'방문객수(명)']]
# costom_output = weather[['방문객수(명)']]

print(weather.shape,customer.shape)


'''
# 결측치 평균으로 채우기
weather_input.loc[weather_input['평균수온'] != weather_input['평균수온'], '평균수온'] = weather_input['평균수온'].mean()
weather_input.loc[weather_input['평균풍속'] != weather_input['평균풍속'], '평균풍속'] = weather_input['평균풍속'].mean()
weather_input.loc[weather_input['평균기압'] != weather_input['평균기압'], '평균기압'] = weather_input['평균기압'].mean()
weather_input.loc[weather_input['평균최대파고'] != weather_input['평균최대파고'], '평균최대파고'] = weather_input['평균최대파고'].mean()
weather_input.loc[weather_input['평균파주기'] != weather_input['평균파주기'], '평균파주기'] = weather_input['평균파주기'].mean()

costom_input.loc[costom_input['평균수온'] != costom_input['평균수온'], '평균수온'] = costom_input['평균수온'].mean()
costom_input.loc[costom_input['평균풍속'] != costom_input['평균풍속'], '평균풍속'] = costom_input['평균풍속'].mean()
costom_input.loc[costom_input['평균기압'] != costom_input['평균기압'], '평균기압'] = costom_input['평균기압'].mean()
costom_input.loc[costom_input['평균최대파고'] != costom_input['평균최대파고'], '평균최대파고'] = costom_input['평균최대파고'].mean()
costom_input.loc[costom_input['평균파주기'] != costom_input['평균파주기'], '평균파주기'] = costom_input['평균파주기'].mean()


weatertrain_x, weatertest_x, weatertrain_y, weatertest_y = train_test_split(weather_input.values, weather_output.values, test_size=0.2)
costrain_x, costest_x, costrain_y, costest_y = train_test_split(costom_input.iloc[:, :-1], costom_input['방문객수(명)'], test_size=0.2)

print(weatertrain_x.shape , weatertest_x.shape)
print(costrain_x.shape , costest_x.shape)




# 강수_관측값 예측을 위한  minmax scaler
weather_scale = MinMaxScaler()
train_x_scaled = weather_scale.fit_transform(train_x)
test_x_scaled = weather_scale.transform(test_x)

print(train_x_scaled.shape,test_x_scaled.shape) #(1294, 9) (324, 9)

train_x=train_x_scaled.reshape(1294, 9, 1)
test_x=test_x_scaled.reshape(324, 9, 1)

print(train_x.shape,test_x.shape) #(1294, 3, 3) (324, 3, 3)


# 2. 모델구성
model=Sequential()
model.add(LSTM(units=128,input_shape=(9,1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2))
model.summary()


# 3. 컴파일 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer='adam')
hist=model.fit(train_x,train_y,epochs=3,batch_size=16,validation_split=0.2, callbacks=[earlyStopping])

# 4. 평가, 예측
loss=model.evaluate(test_x,test_y)
pred_y=model.predict(test_x)


print('loss: ',loss)
print('예상 강수량, 기온,: ', pred_y[-1:])

# loss:  284.1539611816406
# 예상 강수량, 기온,:  [[11.9988165 28.228077 ]]
'''