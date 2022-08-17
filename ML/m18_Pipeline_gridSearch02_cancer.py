import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

pipe=Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())])
model=GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

#3. 컴파일, 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 
print('걸린시간:',np.round(end-start,2))
print('cancer_끝')

# =========================================
# RandomForestClassifier 결과:  0.9473684210526315
# RandomForestClassifier_acc score : 0.9473684210526315

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.9648351648351647
# model.score: 0.9473684210526315
# 걸린시간: 7.3

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.9648351648351647
# model.score: 0.956140350877193


# HalvingGridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits    
# best_score:  0.9833333333333334
# model.score: 0.956140350877193
# 걸린시간: 9.26

# HalvingRandomSearchCV
# Fitting 5 folds for each of 22 candidates, totalling 110 fits
# best_score:  0.961111111111111
# model.score: 0.9473684210526315
# 걸린시간: 4.64

# model.score: 0.9385964912280702 <-pipeline 사용

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# model.score: 0.9385964912280702
# 걸린시간: 99.19 <-pip+grid