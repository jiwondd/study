import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                                  shuffle=True,random_state=123,train_size=0.8)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

kFold=KFold(n_splits=5, shuffle=True,random_state=123)

prameters={'n_estimator':[100],
           'learnig_rate':[0.1],
           'max_depth':[2],
           'gamma':[0],
           'min_child_weight':[100],
           'subsample':[1],
           'colsample_bytree':[0],
           'colsample_bylevel':[0],
           'colsample_bynod':[0],
           'reg_alpha':[10],
           'reg_lambda':[10]}


# 2. 모델
xgb=XGBRegressor(random_state=123)
model=GridSearchCV(xgb,prameters,cv=kFold,n_jobs=8)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가
best_params=model.best_params_
print('최상의 매개변수(파라미터) : ', best_params )
print('최상의 점수 : ',model.best_score_)
result=model.score(x_test,y_test)
print('model.score:',result) 

# 최상의 매개변수(파라미터) :  {'n_estimator': 100}
# 최상의 점수 :  0.20077923073407505
# model.score: 0.46041879678917197

# 최상의 매개변수(파라미터) :  {'learnig_rate': 0.1, 'n_estimator': 100}
# 최상의 점수 :  0.20077923073407505
# model.score: 0.46041879678917197

# 최상의 매개변수(파라미터) :  {'learnig_rate': 0.1, 'max_depth': 2, 'n_estimator': 100}
# 최상의 점수 :  0.2844025412097122
# model.score: 0.5410874145206905

# 최상의 매개변수(파라미터) :  {'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'n_estimator': 100}
# 최상의 점수 :  0.2844025412097122
# model.score: 0.5410874145206905

# 최상의 매개변수(파라미터) :  {'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100}
# 최상의 점수 :  0.38354344380587413
# model.score: 0.5712870345967729

# 최상의 매개변수(파라미터) :  {'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'subsample': 1}
# 최상의 점수 :  0.38354344380587413
# model.score: 0.5712870345967729

# 최상의 매개변수(파라미터) :  {'colsample_bytree': 0, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'subsample': 1}
# 최상의 점수 :  0.39595971106431327
# model.score: 0.5766956673196839

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 0, 'colsample_bytree': 0, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'subsample': 1}
# 최상의 점수 :  0.39595971106431327
# model.score: 0.5766956673196839

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 0, 'colsample_bynod': 0, 'colsample_bytree': 0, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'subsample': 1}
# 최상의 점수 :  0.39595971106431327
# model.score: 0.5766956673196839

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 0, 'colsample_bynod': 0, 'colsample_bytree': 0, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'reg_alpha': 10, 'subsample': 1}
# 최상의 점수 :  0.396399674038559
# model.score: 0.5765859025036232

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 0, 'colsample_bynod': 0, 'colsample_bytree': 0, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 2, 'min_child_weight': 100, 'n_estimator': 100, 'reg_alpha': 10, 'reg_lambda': 10, 'subsample': 1}
# 최상의 점수 :  0.4005155982338452
# model.score: 0.5842708309195914
