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

# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline 
model=make_pipeline(MinMaxScaler(),RandomForestRegressor())

#3. 컴파일, 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

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

# model.score: 0.8065475930348844 <-pipeline

