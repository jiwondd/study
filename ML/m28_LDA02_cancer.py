import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import tensorflow as tf
import pandas as pd

#1. 데이터
datasets=load_breast_cancer()
x = datasets.data 
y = datasets.target

# print(np.unique(y, return_counts=True)) (array([0, 1]), array([212, 357], dtype=int64)) 
# print(y.value_counts()) 판다스 데이터프레임일때는 밸류카운트 하면 됨요

lda=LinearDiscriminantAnalysis(n_components=1) 
lda.fit(x,y)
x=lda.transform(x)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
model=XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
results=model.score(x_test,y_test)
print('결과:',results)
print('걸린시간:',np.round(end-start,2))

# 결과: 0.9824561403508771
# 걸린시간: 0.43