import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
datasets=fetch_covtype()
x = datasets.data 
y = datasets.target

le=LabelEncoder()
y=le.fit_transform(y)

# print(np.unique(y, return_counts=True)) 
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([211840, 283301,  35754,   2747,  
#  9493,  17367,  20510], dtype=int64)) 라벨인코더 해줌


lda=LinearDiscriminantAnalysis(n_components=5) 
lda.fit(x,y)
x=lda.transform(x)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=123)

parameters={'n_estimator':[100,200,300],
           'learnig_rate':[0.1,0.2,0.3,0.5],
           'max_depth':[None,2,3,4,5,6],
           'min_child_weight':[0.1,0.5,1,5],
           'reg_alpha':[0,0.1,1,10],
           'reg_lambda':[0,0.1,1,10]
           }
    

# 2. 모델
xgb=XGBClassifier(random_state=123,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
model=GridSearchCV(xgb,parameters,cv=kFold,n_jobs=8)

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
best_params=model.best_params_
print('최상의 매개변수(파라미터) : ', best_params )
print('최상의 점수 : ',model.best_score_)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('걸린시간 :',np.round(end-start,2))

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 0, 'colsample_bynod': 0, 'colsample_bytree': 0, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'reg_alpha': 10, 'reg_lambda': 10, 'subsample': 1}
# 최상의 점수 :  0.7232992462218419
# model.score: 0.7215906646127896
# 걸린시간 : 9.14