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

x = train_set.drop(['count'], axis=1)
y=train_set['count']

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=777)
# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 분류임 어그로...(논리적인회귀=분류)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
model=LinearSVR()
model1=LinearRegression()
model2=KNeighborsRegressor()
model3=DecisionTreeRegressor()
model4=RandomForestRegressor()

#3.컴파일, 훈련
model.fit(x_train,y_train)
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('LinearSVR의 r2스코어:',r2)
print('LinearSVR 결과: ',result)
print('=========================================')
result=model1.score(x_test,y_test)
y_predict1=model1.predict(x_test)
r2=r2_score(y_test,y_predict1)
print('LinearRegression의 r2스코어:',r2)
print('LinearRegression 결과: ',result)
print('=========================================')
result=model2.score(x_test,y_test)
y_predict2=model2.predict(x_test)
r2=r2_score(y_test,y_predict2)
print('KNeighborsRegressor r2스코어:',r2)
print('KNeighborsRegressor 결과: ',result)
print('=========================================')
result=model3.score(x_test,y_test)
y_predict3=model3.predict(x_test)
r2=r2_score(y_test,y_predict3)
print('DecisionTreeRegressor r2스코어:',r2)
print('DecisionTreeRegressor 결과: ',result)
print('=========================================')
result=model4.score(x_test,y_test)
y_predict4=model4.predict(x_test)
r2=r2_score(y_test,y_predict4)
print('RandomForestRegressor r2스코어:',r2)
print('RandomForestRegressor 결과: ',result)


# loss: 2912.559326171875
# RMSE 53.968133997201804

# LinearSVR의 r2스코어: 0.20989281969434248
# LinearSVR 결과:  0.20989281969434248
# =========================================
# LinearRegression의 r2스코어: 0.21396065866589653
# LinearRegression 결과:  0.21396065866589653
# =========================================
# KNeighborsRegressor r2스코어: 0.45805667815483686
# KNeighborsRegressor 결과:  0.45805667815483686
# =========================================
# DecisionTreeRegressor r2스코어: 0.7837249444636054
# DecisionTreeRegressor 결과:  0.7837249444636054
# =========================================
# RandomForestRegressor r2스코어: 0.8422935093059704
# RandomForestRegressor 결과:  0.8422935093059704
