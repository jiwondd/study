import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical


#.1 데이터
path='./_data/kaggle_bike/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!


#데이트 타임 연/월/일/시 로 컬럼 나누기
train_set['datetime']=pd.to_datetime(train_set['datetime']) #date time 열을 date time 속성으로 변경
#세부 날짜별 정보를 보기 위해 날짜 데이터를 년도, 월, 일, 시간으로 나눠준다.(분,초는 모든값이 0 이므로 추가하지않는다.)
train_set['year']=train_set['datetime'].dt.year
train_set['month']=train_set['datetime'].dt.month
train_set['day']=train_set['datetime'].dt.day
train_set['hour']=train_set['datetime'].dt.hour

#날짜와 시간에 관련된 피쳐에는 datetime, holiday, workingday,year,month,day,hour 이 있다.
#숫자형으로 나오는 holiday,workingday,month,hour만 쓰고 나머지 제거한다.

train_set.drop(['datetime','day','year'],inplace=True,axis=1) #datetime, day, year 제거하기

#month, hour은 범주형으로 변경해주기
train_set['month']=train_set['month'].astype('category')
train_set['hour']=train_set['hour'].astype('category')

#season과 weather은 범주형 피쳐이다. 두 피쳐 모두 숫자로 표현되어 있으니 문자로 변환해준다.
train_set=pd.get_dummies(train_set,columns=['season','weather'])

#casual과 registered는 test데이터에 존재하지 않기에 삭제한다.
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
#temp와 atemp는 상관관계가 아주 높고 두 피쳐의 의미가 비슷하기 때문에 temp만 사용한다.
train_set.drop('atemp',inplace=True,axis=1) #atemp 지우기

#위처럼 test_set도 적용하기
test_set['datetime']=pd.to_datetime(test_set['datetime'])

test_set['month']=test_set['datetime'].dt.month
test_set['hour']=test_set['datetime'].dt.hour

test_set['month']=test_set['month'].astype('category')
test_set['hour']=test_set['hour'].astype('category')

test_set=pd.get_dummies(test_set,columns=['season','weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y=train_set['count']

# print(train_set.shape) #(10886, 16)
# print(test_set.shape) #(6493, 15)
# print(x.shape) #(10886, 15)
# print(y.shape) #(10886,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_set=scaler.transform(test_set)

#2. 모델구성
# model=Sequential()
# model.add(Dense(40,input_dim=15))
# model.add(Dense(60,activation='ReLU'))
# model.add(Dense(100,activation='ReLU'))
# model.add(Dense(50,activation='ReLU'))
# model.add(Dense(30,activation='ReLU'))
# model.add(Dense(10,activation='ReLU'))
# model.add(Dense(1))

# model.save("./_save/keras23_17_save_model_kaggle_house.h5")

# #3. 컴파일, 훈련
# earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) #
# model.compile(loss='mse',optimizer="adam")
# hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
#                epochs=3000, batch_size=100, verbose=1)

# model.save("./_save/keras23_17_save_model_kaggle_house.h5")
model=load_model("./_save/keras23_17_save_model_kaggle_house.h5")

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("RMSE",rmse)

y_summit=model.predict(test_set)


# print(y_summit)
# print(y_summit.shape)


'''
loss: 4995.0439453125
RMSE 70.6756193924409 <-기존

loss: 5038.8125
RMSE 70.9845964680797 <-MinMax

loss: 4983.4833984375
RMSE 70.59379586941036 <-Standard

loss: 4774.572265625
RMSE 69.09828351579266  <-MaxAbsScaler

loss: 5107.462890625
RMSE 71.46650641944817  <-RobustScaler

loss: 5030.474609375
RMSE 70.92583773413457 <- 함수

loss: 4920.2197265625
RMSE 70.14427771302485 <-load_model
'''