from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)
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


# loss: 0.43223631381988525
# r2스코어: 0.6701515707751251 <-랜덤스테이트 42

# LinearSVR의 r2스코어: 0.513207573700792
# LinearSVR 결과:  0.513207573700792
# =========================================
# LinearRegression의 r2스코어: 0.5757877060324512
# LinearRegression 결과:  0.5757877060324512
# =========================================
# KNeighborsRegressor r2스코어: 0.6757890024425293
# KNeighborsRegressor 결과:  0.6757890024425293
# =========================================
# DecisionTreeRegressor r2스코어: 0.6136966873549363
# DecisionTreeRegressor 결과:  0.6136966873549363
# =========================================
# RandomForestRegressor r2스코어: 0.806997225047096
# RandomForestRegressor 결과:  0.806997225047096


