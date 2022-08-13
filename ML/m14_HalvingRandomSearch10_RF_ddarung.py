import numpy as np
import pandas as pd
from pytest import yield_fixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
y=train_set['count']

# x_train,x_test,y_train,y_test=train_test_split(x,y,
#         train_size=0.99,shuffle=True, random_state=1004)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=1004)

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
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model=HalvingRandomSearchCV(RandomForestRegressor(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)


# 3. 컴파일 훈련
import time
start=time.time()
model.fit(x,y)
end=time.time()

print("최적의 매개변수: ",model.best_estimator_)
print("최적의 파라미터: ",model.best_params_)
print("best_score: ",model.best_score_)
print("model.score:",model.score(x,y))

#4. 평가, 예측
y_predict=model.predict(x)
r2=r2_score(y,y_predict)
print('r2 score :', r2)
y_pred_best=model.best_estimator_.predict(x)
print("최적 튠 r2 : ",r2_score(y,y_pred_best))
print('걸린시간:',np.round(end-start,2))

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=23)   
# 최적의 파라미터:  {'max_depth': 23, 'n_estimators': 100}
# best_score:  0.7699900354209397
# model.score: 0.7796024883992262 
# r2 score : 0.7796024883992262   
# 최적 튠 r2 :  0.7796024883992262
# 걸린시간: 12.52

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=10, n_estimators=200)
# 최적의 파라미터:  {'n_estimators': 200, 'max_depth': 10}
# best_score:  0.7691344718664241
# model.score: 0.8300100567508004
# r2 score : 0.8300100567508004
# 최적 튠 r2 :  0.8300100567508004
# 걸린시간: 4.72

# HalvingGridSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 48
# max_resources_: 1314
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 48
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 144
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 432
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 1296
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=5, min_samples_split=5, n_jobs=4)
# 최적의 파라미터:  {'min_samples_leaf': 5, 'min_samples_split': 5, 'n_jobs': 4} 
# best_score:  0.7559535223855731
# model.score: 0.7890963265616713
# r2 score : 0.7890963265616713
# 최적 튠 r2 :  0.7890963265616713
# 걸린시간: 9.23

# HalvingRandomSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 5
# min_resources_: 10
# max_resources_: 1328
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 10
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# C:\Anaconda3\envs\tf282gpu\lib\site-packages\sklearn\metrics\_regression.py:682: UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
#   warnings.warn(msg, UndefinedMetricWarning)
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 30
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 90
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 270
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=10, n_jobs=2)
# 최적의 파라미터:  {'n_jobs': 2, 'min_samples_split': 10, 'min_samples_leaf': 3}
# best_score:  0.6429836496336111
# model.score: 0.9100815624532531
# r2 score : 0.9100815624532531
# 최적 튠 r2 :  0.9100815624532531
# 걸린시간: 8.59