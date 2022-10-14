import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.covariance import EllipticEnvelope

#.1 데이터
path='./_data/kaggle_bike/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!

# print(train_set.shape) #(10886, 12)
# print(test_set.shape) #(6493, 9)

train_set['datetime']=pd.to_datetime(train_set['datetime'])
train_set['year']=train_set['datetime'].dt.year
train_set['month']=train_set['datetime'].dt.month
train_set['day']=train_set['datetime'].dt.day
train_set['hour']=train_set['datetime'].dt.hour
train_set.drop(['datetime','day','year'],inplace=True,axis=1)

train_set['month']=train_set['month'].astype('category')
train_set['hour']=train_set['hour'].astype('category')

train_set=pd.get_dummies(train_set,columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp',inplace=True,axis=1)

test_set['datetime']=pd.to_datetime(test_set['datetime'])
test_set['month']=test_set['datetime'].dt.month
test_set['hour']=test_set['datetime'].dt.hour
test_set['month']=test_set['month'].astype('category')
test_set['hour']=test_set['hour'].astype('category')

test_set=pd.get_dummies(test_set,columns=['season','weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count','humidity'], axis=1)
y =train_set['count']

outliers=EllipticEnvelope(contamination=.2) 
outliers.fit(x)
outliers1=outliers.predict(x)
print(outliers1)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=123)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor

model=XGBRegressor()

model.fit(x_train,y_train)

results=model.score(x_test,y_test)
print('결과:',results)

# 결과: 0.8791807984062061
