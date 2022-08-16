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


# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline 
model=make_pipeline(MinMaxScaler(),RandomForestRegressor())

#3. 컴파일, 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

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

# model.score: 0.9013278194768353 <-pipeline