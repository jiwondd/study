import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=72)

# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline 
model=make_pipeline(MinMaxScaler(),RandomForestRegressor())

#3. 컴파일, 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 



# loss: 17.545747756958008
# r2스코어: 0.7876257019581974

# load_boston
# LinearSVR의 r2스코어: 0.688808809413431
# LinearSVR 결과:  0.688808809413431
# =========================================
# LinearRegression의 r2스코어: 0.7000498170510017
# LinearRegression 결과:  0.7000498170510017
# =========================================
# KNeighborsRegressor r2스코어: 0.7209948731531117
# KNeighborsRegressor 결과:  0.7209948731531117
# =========================================
# DecisionTreeRegressor r2스코어: 0.8276211027252549
# DecisionTreeRegressor 결과:  0.8276211027252549
# =========================================
# RandomForestRegressor r2스코어: 0.8595099995845223
# RandomForestRegressor 결과:  0.8595099995845223

# model.score: 0.7636348988993157 <- pipeline
