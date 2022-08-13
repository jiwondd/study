
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
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

n_splits=5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)

parameters=[
    {'n_estimators':[100,200],'max_depth':[6,8,10,23]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],
     'n_jobs':[-1,2,4]},
]

#2. 모델구성
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
model=HalvingRandomSearchCV(RandomForestClassifier(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

#3. 컴파일, 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

print("최적의 매개변수: ",model.best_estimator_)
print("최적의 파라미터: ",model.best_params_)
print("best_score: ",model.best_score_)
print("model.score:",model.score(x_test,y_test))

#4. 평가, 예측
y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
y_pred_best=model.best_estimator_.predict(x_test)
print("최적 튠 acc : ",accuracy_score(y_test,y_pred_best))
print('걸린시간:',np.round(end-start,2))

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=3, min_samples_split=3, n_jobs=4)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 3, 'n_jobs': 4}
# best_score:  0.938579073901494
# model.score: 0.9440117725015705
# acc score : 0.9440117725015705
# 최적 튠 acc :  0.9440117725015705
# 걸린시간: 4537.92

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=3, min_samples_split=3, n_jobs=4)
# 최적의 파라미터:  {'n_jobs': 4, 'min_samples_split': 3, 'min_samples_leaf': 3}
# best_score:  0.9381358803007815
# model.score: 0.9441666738380249
# acc score : 0.9441666738380249
# 최적 튠 acc :  0.9441666738380249
# 걸린시간: 858.69

# HalvingRandomSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 9
# min_resources_: 70
# max_resources_: 464809
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 70
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 210
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 630
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 1890
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestClassifier(max_depth=23, n_estimators=200)
# 최적의 파라미터:  {'n_estimators': 200, 'max_depth': 23}
# best_score:  0.7501354328940536
# model.score: 0.9179367141984286
# acc score : 0.9179367141984286
# 최적 튠 acc :  0.9179367141984286
# 걸린시간: 200.03
# PS C:\study> 