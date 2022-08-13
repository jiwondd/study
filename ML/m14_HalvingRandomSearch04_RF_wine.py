import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

n_splits=5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)

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

model=HalvingRandomSearchCV(RandomForestClassifier(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)

#3. 컴파일, 훈련
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
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
y_pred_best=model.best_estimator_.predict(x_test)
print("최적 튠 acc : ",accuracy_score(y_test,y_pred_best))
print('걸린시간:',np.round(end-start,2))


# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestClassifier(max_depth=8)
# 최적의 파라미터:  {'max_depth': 8, 'n_estimators': 100}
# best_score:  0.9859605911330049
# model.score: 1.0
# acc score : 1.0
# 최적 튠 acc :  1.0
# 걸린시간: 6.36

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=3, min_samples_split=10, n_jobs=2)
# 최적의 파라미터:  {'n_jobs': 2, 'min_samples_split': 10, 'min_samples_leaf': 3}
# best_score:  0.9788177339901478
# model.score: 0.9722222222222222
# acc score : 0.9722222222222222
# 최적 튠 acc :  0.9722222222222222
# 걸린시간: 2.49

# HalvingGridSearchCV
# n_iterations: 2
# n_required_iterations: 4
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 142
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 30
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 90
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# 최적의 매개변수:  RandomForestClassifier(max_depth=8)
# 최적의 파라미터:  {'max_depth': 8, 'n_estimators': 100}
# best_score:  0.977124183006536
# model.score: 1.0
# acc score : 1.0
# 최적 튠 acc :  1.0
# 걸린시간: 8.22

# HalvingRandomSearchCV
# n_iterations: 2
# n_required_iterations: 2
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 142
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 4
# n_resources: 30
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 1
# n_candidates: 2
# n_resources: 90
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수:  RandomForestClassifier(max_depth=8)
# 최적의 파라미터:  {'n_estimators': 100, 'max_depth': 8}
# best_score:  0.9882352941176471
# model.score: 1.0
# acc score : 1.0
# 최적 튠 acc :  1.0
# 걸린시간: 2.13

