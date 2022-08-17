from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)


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
print('california_끝')

# =========================================
# RandomForestRegressor r2스코어: 0.806997225047096
# RandomForestRegressor 결과:  0.806997225047096

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.8045142793956075
# model.score: 0.8071124646751686
# 걸린시간: 199.11

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.803400754455893
# model.score: 0.804976136795321
# 걸린시간: 35.4

# HalvingGridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# best_score:  0.8040950688797361
# model.score: 0.8025474280002083
# 걸린시간: 35.76

# HalvingRandomSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# best_score:  0.589184095277874
# model.score: 0.8640126315988368
# 걸린시간: 19.03

# model.score: 0.8065475930348844 <-pipeline

# pip+grid
# Fitting 5 folds for each of 56 candidates, totalling 280 fits 2.0\pyt
# model.score: 0.8072419427049692                               1544' '
# 걸린시간: 610.12