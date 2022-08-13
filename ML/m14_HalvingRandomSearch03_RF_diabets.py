from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        test_size=0.15,shuffle=True, random_state=42)
# C:\Anaconda3\envs\tf282gpu\lib\site-packages\sklearn\metrics\_regression.py:682: UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
#   warnings.warn(msg, UndefinedMetricWarning) 데이터가 너무 적어서 스코어가 안나와! 테스트사이즈를 조정해봐!

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=777)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

parameters=[
    {'n_estimators':[100,200],'max_depth':[6,8,10,23]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],
     'n_jobs':[-1,2,4]},
]

#2. 모델구성
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model=HalvingRandomSearchCV(RandomForestRegressor(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)


# 3. 컴파일 훈련
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
r2=r2_score(y_test,y_predict)
print('r2 score :', r2)
y_pred_best=model.best_estimator_.predict(x_test)
print("최적 튠 r2 : ",r2_score(y_test,y_pred_best))
print('걸린시간:',np.round(end-start,2))


# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, n_jobs=4)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 2, 'n_jobs': 4}
# best_score:  0.46146949009288657
# model.score: 0.38710381659703685
# r2 score : 0.38710381659703685  
# 최적 튠 r2 :  0.38710381659703696
# 걸린시간: 6.94

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=10, n_jobs=4)
# 최적의 파라미터:  {'n_jobs': 4, 'min_samples_split': 10, 'min_samples_leaf': 3}
# best_score:  0.4549962129337361
# model.score: 0.40742604815261896
# r2 score : 0.40742604815261907
# 최적 튠 r2 :  0.40742604815261907
# 걸린시간: 2.57

# HalvingGridSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 11
# max_resources_: 309
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 11
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 33
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 99
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 297
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=3, n_jobs=4)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 3, 'n_jobs': 4}best_score:  0.4501968986897034
# model.score: 0.38869794371618804
# r2 score : 0.38869794371618804
# 최적 튠 r2 :  0.38869794371618804
# 걸린시간: 8.19

# HalvingRandomSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 10
# max_resources_: 375
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 37
# n_resources: 10
# Fitting 5 folds for each of 37 candidates, totalling 185 fits
# ----------
# iter: 1
# n_candidates: 13
# n_resources: 30
# Fitting 5 folds for each of 13 candidates, totalling 65 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 90
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 270
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=5, min_samples_split=10, n_jobs=2)
# 최적의 파라미터:  {'n_jobs': 2, 'min_samples_split': 10, 'min_samples_leaf': 5}
# best_score:  0.47705236723363964
# model.score: 0.4380044687706238
# r2 score : 0.4380044687706238
# 최적 튠 r2 :  0.4380044687706238
# 걸린시간: 5.79