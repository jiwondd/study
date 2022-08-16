from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

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


# RandomForestRegressor r2스코어: 0.5681461196489572
# RandomForestRegressor 결과:  0.5681461196489572

# model.score: 0.5718076805473089 <-pipeline