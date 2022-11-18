#https://www.kaggle.com/competitions/bike-sharing-demand/submit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from icecream import ic
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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

x = train_set.drop(['count','humidity'], axis=1)
y=train_set['count']

# print(x.shape) #(10886, 15) ->(10886, 14)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=KFold(n_splits=5,shuffle=True,random_state=123)

mm=MinMaxScaler() 
stan=StandardScaler()
ma=MaxAbsScaler()
robus=RobustScaler() 
quan=QuantileTransformer()
power_yeo=PowerTransformer(method='yeo-johnson') 
power_box=PowerTransformer(method='box-cox')

scalers=[mm,stan,ma,robus,quan,power_yeo,power_box]
for scaler in scalers:
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model=RandomForestRegressor()
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        result=r2_score(y_test,y_predict)
        scale_name=scaler.__class__.__name__
        print('{0}결과:{1:4f}'.format(scale_name,result))
        
# MinMaxScaler결과:0.816020
# StandardScaler결과:0.814775
# MaxAbsScaler결과:0.814196
# RobustScaler결과:0.814731
# QuantileTransformer결과:0.814259
# PowerTransformer결과:0.815052
# The Box-Cox transformation can only be applied to strictly positive data