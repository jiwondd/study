from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=42)

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
from sklearn.model_selection import HalvingGridSearchCV

model=HalvingGridSearchCV(RandomForestRegressor(),parameters,cv=kfold, verbose=1,
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

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=23, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 23, 'n_estimators': 200}
# best_score:  0.8045142793956075
# model.score: 0.8071124646751686
# r2 score : 0.8071124646751686
# 최적 튠 r2 :  0.8071124646751686
# 걸린시간: 199.11

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=5, n_jobs=4)
# 최적의 파라미터:  {'n_jobs': 4, 'min_samples_split': 5, 'min_samples_leaf': 3}
# best_score:  0.803400754455893
# model.score: 0.804976136795321
# r2 score : 0.804976136795321
# 최적 튠 r2 :  0.804976136795321
# 걸린시간: 35.4

# HalvingGridSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 611
# max_resources_: 16512
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 611
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 1833
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 5499
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 16497
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, n_jobs=-1)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 2, 'n_jobs': -1}
# best_score:  0.8040950688797361
# model.score: 0.8025474280002083
# r2 score : 0.8025474280002084
# 최적 튠 r2 :  0.8025474280002084
# 걸린시간: 35.76