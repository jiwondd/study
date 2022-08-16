
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=fetch_covtype()
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
print('페치코브끝')

# loss: 0.19975309073925018
# accuracy: 0.920501172542572
# ===================================
# acc score : 1.0<-RobustScaler

# LinearSVC 결과:  0.711220880700154
# LinearSVC score : 0.711220880700154
# =========================================
# LogisticRegression 결과:  0.7222963262566371
# LogisticRegression_acc score : 0.7222963262566371
# =========================================
# KNeighborsClassifier 결과:  0.9273856957221415
# KNeighborsClassifier_acc score : 0.9273856957221415
# =========================================
# DecisionTreeClassifier 결과:  0.9385730144660637
# DecisionTreeClassifier_acc score : 0.9385730144660637
# =========================================
# RandomForestClassifier 결과:  0.9555433164376135
# RandomForestClassifier_acc score : 0.9555433164376135

# model.score: 0.9556551896250527 <- pipeline
