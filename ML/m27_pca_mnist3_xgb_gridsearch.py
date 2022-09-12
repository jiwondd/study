from inspect import Parameter
from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.datasets import mnist
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier

# 1. 데이터
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)

pca=PCA(n_components=486)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

x_train, x_test, y_train, y_test=train_test_split(x_train,y_train,train_size=0.8,stratify=y_train,
                                                  random_state=123,shuffle=True)

n_splits=713
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

Parameter=[
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.3,0.001,0.01],"max_depth":[4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1,0.001,0.01],"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110],"learning_rate":[0.1,0.001,0.5],"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],"colsample_bylevel":[0.6,0.7,0.9]}
]

# 2. 모델구성
model=RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0),Parameter,cv=kfold, verbose=1,
                   refit=True,n_jobs=-1)

# 3. 컴파일 훈련
import time
start=time.time()
model.fit(x_train,y_train,verbose=2) 
end=time.time()

# 4. 평가 예측
result=model.score(x_test,y_test)
print('결과:',result)
print('걸린시간 :',np.round(end-start,2))

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 결과: 0.9601666666666666
# 걸린시간 : 772.14  