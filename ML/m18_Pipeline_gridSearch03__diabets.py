from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

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
print('diabet_끝')


# =========================================
# RandomForestRegressor r2스코어: 0.5681461196489572
# RandomForestRegressor 결과:  0.5681461196489572

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.46146949009288657
# model.score: 0.38710381659703685
# 걸린시간: 6.94

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.4549962129337361
# model.score: 0.40742604815261896
# 걸린시간: 2.57

# HalvingGridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# model.score: 0.38869794371618804
# 걸린시간: 8.19

# HalvingRandomSearchCV
# Fitting 5 folds for each of 37 candidates, totalling 185 fits
# Fitting 5 folds for each of 13 candidates, totalling 65 fits
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# best_score:  0.47705236723363964
# model.score: 0.4380044687706238
# 걸린시간: 5.79

# model.score: 0.5718076805473089 <-pipeline

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# model.score: 0.5868650856896341
# 걸린시간: 98.01 <-pip+grid