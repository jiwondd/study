import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        test_size=0.15,shuffle=True, random_state=777)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=777)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)

parameters=[
    {'n_estimators':[100,200],'max_depth':[6,8,10,23]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],
     'n_jobs':[-1,2,4]},
]

#2. 모델구성
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV,HalvingRandomSearchCV

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
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=3, n_jobs=-1)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 3, 'n_jobs': -1}
# best_score:  0.7853511748016508
# model.score: 0.8189322087487194
# r2 score : 0.8189322087487194
# 최적 튠 r2 :  0.8189322087487195
# 걸린시간: 7.56

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=23)
# 최적의 파라미터:  {'n_estimators': 100, 'max_depth': 23}
# best_score:  0.7847039298898398
# model.score: 0.8563841478061387
# r2 score : 0.8563841478061387
# 최적 튠 r2 :  0.8563841478061387
# 걸린시간: 2.71

# HalvingGridSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 13
# max_resources_: 354
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 13
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 39
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 117
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 351
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=23, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 23, 'n_estimators': 200}
# best_score:  0.7841623330565938
# model.score: 0.854932396348566
# r2 score : 0.854932396348566
# 최적 튠 r2 :  0.854932396348566
# 걸린시간: 8.92

# HalvingRandomSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 10
# max_resources_: 430
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 43
# n_resources: 10
# Fitting 5 folds for each of 43 candidates, totalling 215 fits
# ----------
# iter: 1
# n_candidates: 15
# n_resources: 30
# Fitting 5 folds for each of 15 candidates, totalling 75 fits
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
# 최적의 매개변수:  RandomForestRegressor(max_depth=10, n_estimators=200)
# 최적의 파라미터:  {'n_estimators': 200, 'max_depth': 10}
# best_score:  0.8328550167864173
# model.score: 0.8699657313301827
# r2 score : 0.8699657313301827
# 최적 튠 r2 :  0.8699657313301827
# 걸린시간: 6.97