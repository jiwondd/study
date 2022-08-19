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

x = np.delete(x,[1,3,5],axis=1)
# print(x.shape) (178, 13)->(178, 10)

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

# DecisionTreeClassifier() 의 model.score: 0.7222222222222222
# DecisionTreeClassifier() 의 acc_score : 0.7222222222222222
# DecisionTreeClassifier() : [0.02086426 0.         0.0399355  0.         0.02087296 0.
#  0.06138924 0.         0.         0.3228603  0.         0.11183169
#  0.42224605]
# *************************************************
# RandomForestClassifier() 의 model.score: 1.0
# RandomForestClassifier() 의 acc_score : 1.0
# RandomForestClassifier() : [0.15237366 0.02880526 0.0130605  0.01477161 0.04195943 0.04522411
#  0.19881961 0.01440677 0.01837399 0.12486276 0.0603598  0.12185386
#  0.16512864]
# *************************************************
# GradientBoostingClassifier() 의 model.score: 0.9444444444444444
# GradientBoostingClassifier() 의 acc_score : 0.9444444444444444
# GradientBoostingClassifier() : [0.05506804 0.00899412 0.01328399 0.00340046 0.04041746 0.0207263
#  0.28267397 0.00036177 0.00140976 0.23447475 0.04402667 0.0305432
#  0.2646195 ]
# *************************************************
# XGBClassifier의 model.score: 0.9722222222222222
# XGBClassifier의 acc_score : 0.9722222222222222
# XGBClassifier: [0.0824663  0.01322483 0.02392327 0.00823698 0.08485855 0.10585225855 0.10585225
#  0.2242181  0.00355626 0.01219124 0.14172232 0.03515361 0.0229305
#  0.24166587]

# [1,3,5] 컬럼제거 / DecisionTreeClassifier는 성능이 확연히 안좋아졌고 나머지는 비슷함
# DecisionTreeClassifier() 의 model.score: 0.6944444444444444
# DecisionTreeClassifier() 의 acc_score : 0.6944444444444444
# DecisionTreeClassifier() : [0.02086426 0.0399355  0.         0.04313301 0.    
#      0.0159742
#  0.34111653 0.02087296 0.11183169 0.40627185]
# *************************************************
# RandomForestClassifier() 의 model.score: 1.0
# RandomForestClassifier() 의 acc_score : 1.0
# RandomForestClassifier() : [0.17831976 0.02431146 0.02869717 0.18897235 0.00977045 0.02281248
#  0.16976128 0.08953469 0.11556101 0.17225934]
# *************************************************
# GradientBoostingClassifier() 의 model.score: 0.9444444444444444
# GradientBoostingClassifier() 의 acc_score : 0.9444444444444444
# GradientBoostingClassifier() : [5.30367606e-02 1.70762276e-02 4.48191856e-02 2.71437597e-01
#  2.59141801e-04 1.98793112e-03 2.33997279e-01 4.76404750e-02
#  6.48457242e-02 2.64899678e-01]
# *************************************************
# XGBClassifier의 model.score: 0.9722222222222222
# XGBClassifier의 acc_score : 0.9722222222222222
# XGBClassifier: [0.11676899 0.02936585 0.09841432 0.24370252 0.00484352 0.01320541
#  0.16470593 0.04366479 0.02797465 0.25735405]
