from concurrent.futures import thread
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings(action='ignore')
import xgboost as xg
from sklearn.linear_model import LinearRegression

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target
print(x.shape,y.shape) #(442, 10) (442,)
print(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                                  shuffle=True,random_state=72,train_size=0.8)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# kFold=KFold(n_splits=5, shuffle=True,random_state=123)

# 2. 모델
# model=XGBRegressor(random_state=100,
#                     n_estimators=100,
#                     learning_rate=0.3,
#                     max_depth=6,
#                     gamma=0)
model=LinearRegression()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가
result=model.score(x_test,y_test)
print('model.score:',result) 
y_predict=model.predict(x_test)
r2=r2_score(y_test, y_predict)
print('진짜 최종 test 점수 : ' , r2)

# model.score: -1.5732111511642377
# 진짜 최종 test 점수 :  -1.5732111511642377
# [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819        
#  0.06012432 0.09595273 0.30483875 0.06629313]

# thresholds=model.feature_importances_
# print('--------------------------------------')

# for thresh in thresholds :
#     selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
#     select_x_train=selection.transform(x_train)
#     select_x_test=selection.transform(x_test)
#     print(select_x_train.shape,select_x_test.shape)
    
#     selection_model=XGBRegressor(n_jobs=-1,
#                                  random_state=100,
#                                  n_estimators=100,
#                                  learning_rate=0.3,
#                                  max_depth=6,
#                                  gamma=0)
#     selection_model.fit(select_x_train,y_train)
#     y_predict=selection_model.predict(select_x_test)
#     score=r2_score(y_test,y_predict)
#     print("Thresh=%.3f,n=%d, R2:%.2f%%"
#           #소수점3개까지,정수,소수점2개까지
#           %(thresh,select_x_train.shape[1],score*100))
    
# print(xg.__version__)

# model.score: 0.5675916622351824 randomstate 123
# model.score: 0.6579209558684549 randomstate 72
    