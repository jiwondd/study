#https://www.kaggle.com/competitions/bike-sharing-demand/submit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

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

print(x.shape) #(10886, 15) ->(10886, 14)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=777)
# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model1=DecisionTreeRegressor()
model2=RandomForestRegressor()
model3=GradientBoostingRegressor()
model4=XGBRegressor()

# 3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4.평가, 예측
result1=model1.score(x_test,y_test)
print(model1,'의 model.score:',result1)
y_predict1=model1.predict(x_test)
r21=r2_score(y_test,y_predict1)
print(model1,'의 r2_score :',r21)
print(model1,':',model1.feature_importances_)
print('*************************************************')
result2=model2.score(x_test,y_test)
print(model2,'의 model.score:',result2)
y_predict2=model2.predict(x_test)
r22=r2_score(y_test,y_predict2)
print(model2,'의 r2_score :',r22)
print(model2,':',model2.feature_importances_)
print('*************************************************')
result3=model3.score(x_test,y_test)
print(model3,'의 model.score:',result3)
y_predict3=model3.predict(x_test)
r23=r2_score(y_test,y_predict3)
print(model3,'의 r2_score :',r23)
print(model3,':',model3.feature_importances_)
print('*************************************************')
result4=model4.score(x_test,y_test)
print(model4,'의 model.score:',result4)
y_predict4=model4.predict(x_test)
r24=r2_score(y_test,y_predict4)
print(model4,'의 r2_score :',r24)
print(model4,':',model4.feature_importances_)
print('*************************************************')

# DecisionTreeRegressor() 의 model.score: 0.7411765689150476
# DecisionTreeRegressor() 의 r2_score : 0.7411765689150476
# DecisionTreeRegressor() : [3.21686135e-03 7.80985560e-02 1.32043637e-01 7.15308416e-02   
#  3.65245431e-02 4.83663365e-02 5.99700955e-01 1.00719021e-03
#  3.01660068e-03 1.83743433e-03 2.10113998e-03 3.92186999e-03
#  2.92792309e-03 1.57059087e-02 2.03053152e-07]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.8451829567539365
# RandomForestRegressor() 의 r2_score : 0.8451829567539365
# RandomForestRegressor() : [3.17253646e-03 6.66316542e-02 1.40764049e-01 7.18319869e-02
#  3.51470200e-02 4.11487109e-02 6.04328066e-01 7.21734596e-03
#  2.49027930e-03 2.19470639e-03 3.24566227e-03 4.66567246e-03
#  3.30665715e-03 1.38551789e-02 4.74756766e-07]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.7701710523437547
# GradientBoostingRegressor() 의 r2_score : 0.7701710523437547
# GradientBoostingRegressor() : [3.09596730e-04 1.16105342e-01 1.25332565e-01 3.47748006e-02
#  2.47346684e-03 3.17556486e-02 6.66000396e-01 5.64045938e-03
#  4.54652622e-04 1.98481123e-04 2.54534533e-03 1.08903281e-03
#  8.24041658e-05 1.32378094e-02 0.00000000e+00]
# *************************************************
# XGBRegressor의 model.score: 0.8492578962711205
# XGBRegressor의 r2_score : 0.8492578962711205
# XGBRegressor: [0.02035182 0.19447969 0.0801827  0.03008506 0.01172729 0.05399299
#  0.38874286 0.         0.01276062 0.01285707 0.         0.02840187
#  0.01017065 0.1562474  0.        ]

# [7,10] 컬럼제거 후 / 전반적으로 성능이 하락했다.
# DecisionTreeRegressor() 의 model.score: 0.6911137056610303
# DecisionTreeRegressor() 의 r2_score : 0.6911137056610303
# DecisionTreeRegressor() : [4.09740893e-03 7.73897719e-02 1.53825607e-01 5.31770160e-02   
#  3.03551674e-02 6.08190767e-01 2.27267980e-02 4.28785605e-03
#  4.65847307e-03 2.93846620e-03 9.02038115e-03 5.77073206e-03
#  2.35607884e-02 7.66323313e-07]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.8203731461916683
# RandomForestRegressor() 의 r2_score : 0.8203731461916683
# RandomForestRegressor() : [4.39344941e-03 6.77625311e-02 1.60156512e-01 5.25533124e-02
#  4.49982997e-02 6.11945457e-01 7.70344850e-03 3.75745823e-03
#  3.72967978e-03 5.09266380e-03 9.09198676e-03 5.42133087e-03
#  2.33933343e-02 5.36293715e-07]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.7882557499281706
# GradientBoostingRegressor() 의 r2_score : 0.7882557499281706
# GradientBoostingRegressor() : [3.28037454e-04 1.09849004e-01 1.32410074e-01 1.57356298e-03
#  2.49212671e-02 6.94189072e-01 1.02055853e-02 3.88318128e-04
#  5.11022962e-04 2.57869818e-03 4.00400255e-03 3.27362976e-04
#  1.87139927e-02 0.00000000e+00]
# *************************************************
# XGBRegressor의 model.score: 0.853991352012587
# XGBRegressor의 r2_score : 0.853991352012587
# XGBRegressor: [0.02651464 0.1970494  0.07960898 0.01157842 0.04600016 0.3881419
#  0.         0.01785406 0.02608532 0.         0.03429582 0.01225743
#  0.16061376 0.        ]