import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
y=train_set['count']

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)

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


# loss: 722.6207885742188
# RMSE 26.881613397126884

# LinearSVR의 r2스코어: 0.7549003140609883
# LinearSVR 결과:  0.7549003140609883
# =========================================
# LinearRegression의 r2스코어: 0.7981674407799786
# LinearRegression 결과:  0.7981674407799786
# =========================================
# KNeighborsRegressor r2스코어: 0.8650296541055397
# KNeighborsRegressor 결과:  0.8650296541055397
# =========================================
# DecisionTreeRegressor r2스코어: 0.887491996429356
# DecisionTreeRegressor 결과:  0.887491996429356
# =========================================
# RandomForestRegressor r2스코어: 0.8947086700285558
# RandomForestRegressor 결과:  0.8947086700285558