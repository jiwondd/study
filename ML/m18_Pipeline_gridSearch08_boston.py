import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=72)

parameters=[
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,23]},
    {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[2,3,5,10],
     'RF__n_jobs':[-1,2,4]}
]

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=1004)

#2. 모델구성
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

pipe=Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestRegressor())])
model=GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

#3. 컴파일, 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 
print('걸린시간:',np.round(end-start,2))
print('boston_끝')

# =========================================
# RandomForestRegressor r2스코어: 0.8595099995845223
# RandomForestRegressor 결과:  0.8595099995845223

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.7853511748016508
# model.score: 0.8189322087487194
# 걸린시간: 7.56

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.7847039298898398
# model.score: 0.8563841478061387
# 걸린시간: 2.71

# HalvingGridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# best_score:  0.7841623330565938
# model.score: 0.854932396348566
# 걸린시간: 8.92

# HalvingRandomSearchCV
# Fitting 5 folds for each of 43 candidates, totalling 215 fits
# Fitting 5 folds for each of 15 candidates, totalling 75 fits
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# best_score:  0.8328550167864173
# model.score: 0.8699657313301827
# 걸린시간: 6.97


# model.score: 0.7636348988993157 <- pipeline

# pip+grid
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# model.score: 0.7721558244711375
# 걸린시간: 95.78 <-pip+gird