import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

parameters=[
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,23]},
    {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[2,3,5,10],
     'RF__n_jobs':[-1,2,4]}
]
from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1004)

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
print('wine_끝')

# RandomForestClassifier 결과:  1.0
# RandomForestClassifier_acc score : 1.0

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.9859605911330049
# model.score: 1.0
# 걸린시간: 6.36

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.9788177339901478
# model.score: 0.9722222222222222
# 걸린시간: 2.49

# HalvingGridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# best_score:  0.977124183006536
# model.score: 1.0
# 걸린시간: 8.22

# HalvingRandomSearchCV
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# best_score:  0.9882352941176471
# model.score: 1.0
# 걸린시간: 2.13

# model.score: 1.0 <-pipeline

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# model.score: 0.9384055555555556
# 걸린시간: 90.55 <- pip+grid
