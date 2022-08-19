
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

# x = np.delete(x,[1,3,5],axis=1)
# print(x.shape) #(581012, 54)

'''
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model1=DecisionTreeClassifier()
model2=RandomForestClassifier()
model3=GradientBoostingClassifier()
model4=XGBClassifier()

# 3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4.평가, 예측
result1=model1.score(x_test,y_test)
print(model1,'의 model.score:',result1)
y_predict1=model1.predict(x_test)
acc1=accuracy_score(y_test,y_predict1)
print(model1,'의 acc_score :',acc1)
print(model1,':',model1.feature_importances_)
print('*************************************************')
result2=model2.score(x_test,y_test)
print(model2,'의 model.score:',result2)
y_predict2=model2.predict(x_test)
acc2=accuracy_score(y_test,y_predict2)
print(model2,'의 acc_score :',acc2)
print(model2,':',model2.feature_importances_)
print('*************************************************')
result3=model3.score(x_test,y_test)
print(model3,'의 model.score:',result3)
y_predict3=model3.predict(x_test)
acc3=accuracy_score(y_test,y_predict3)
print(model3,'의 acc_score :',acc3)
print(model3,':',model3.feature_importances_)
print('*************************************************')
result4=model4.score(x_test,y_test)
print(model4,'의 model.score:',result4)
y_predict4=model4.predict(x_test)
acc4=accuracy_score(y_test,y_predict4)
print(model4,'의 acc_score :',acc4)
print(model4,':',model4.feature_importances_)
print('*************************************************')
'''