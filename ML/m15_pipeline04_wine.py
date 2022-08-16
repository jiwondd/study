import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline 
model=make_pipeline(MinMaxScaler(),RandomForestClassifier())

#3. 컴파일, 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

# loss: 0.15575915575027466
# accuracy: 0.9722222089767456
# ===================================        
# acc score : 1.0   <-RobustScaler

# LinearSVC 결과:  0.9722222222222222
# LinearSVC score : 0.9722222222222222
# =========================================
# LogisticRegression 결과:  0.9722222222222222
# LogisticRegression_acc score : 0.9722222222222222
# =========================================
# KNeighborsClassifier 결과:  0.8333333333333334
# KNeighborsClassifier_acc score : 0.8333333333333334
# =========================================
# DecisionTreeClassifier 결과:  0.7777777777777778
# DecisionTreeClassifier_acc score : 0.7777777777777778
# =========================================
# RandomForestClassifier 결과:  1.0
# RandomForestClassifier_acc score : 1.0

# model.score: 1.0 <-pipeline