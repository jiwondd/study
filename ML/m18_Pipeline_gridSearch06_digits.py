from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical
import sklearn as sk


#1. 데이터
datasets=load_digits()
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
print('digits_끝')

# =========================================
# RandomForestClassifier 결과:  0.9722222222222222
# RandomForestClassifier_acc score : 0.9722222222222222

# model.score: 0.9777777777777777 <-pipeline

# GridtearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.9721544715447153
# model.score: 0.9777777777777777
# 걸린시간: 10.64

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.9728561749903213
# model.score: 0.9722222222222222
# 걸린시간: 3.33

# HalvingGridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# best_score:  0.960949720670391
# model.score: 0.9722222222222222
# 걸린시간: 11.19

# HalvingRandomSearchCV
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# best_score:  0.9576288019863439
# model.score: 0.9805555555555555
# 걸린시간: 4.46

# pip
# Fitting 5 folds for each of 56 candidates, totalling 280 fits     
# model.score: 0.8653518254639134
# 걸린시간: 161.59
# =========================================
# RandomForestClassifier 결과:  0.9722222222222222
# RandomForestClassifier_acc score : 0.9722222222222222

# model.score: 0.9777777777777777 <-pipeline

# pip+grid
# Fitting 5 folds for each of 56 candidates, totalling 280 fits     
# model.score: 0.8653518254639134
# 걸린시간: 161.59

