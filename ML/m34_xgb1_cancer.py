import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target
print(x.shape,y.shape) #569, 30) (569,)

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                                  shuffle=True,random_state=123,train_size=0.8,stratify=y)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=123)

# 'n_estimator':[100,200,300,400,50,1000] 디폴트 100 / 1~무한대 / 정수형태 /에포
# 'learnig_rate':[0.1,0.2,0.3,0.5,1,0.01,0.001] 디폴트 0.3 / 0~1 =eta 
# 'max_depth':[None,2,3,4,5,6,7,8,9,10] 디폴트 6 / 0~무한대 / 정수형태 /깊이 
# 'gamma':[0,1,2,3,4,5,10,100] 디폴트 0 / 0~무한대 
# 'min_child_weight':[0,0.01,0.001,0.1,0.5,1,5,10,100]  / 디폴트 1 / 0~무한대
# 'subsample':[0,0.1,0.2,0.3,0.5,0.6,0.7,1] / 디폴트 1/ 0~1
# 'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.6,0.7,1] / 디폴트 1 / 0~1
# 'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.6,0.7,1] / 디폴트 1 / 0~1
# 'colsample_bynod':[0,0.1,0.2,0.3,0.5,0.6,0.7,1] / 디폴트 1 / 0~1
# 'reg_alpha':[0.1,0,0.001,0.3,1,2,10] / 디폴트 0 / 0~무한대 / L1 절대값 가중치 규제 / =alpha
# 'reg_lambda':[0.1,0,0.001,0.3,1,2,10] / 디폴트 1 / 0~무한대 / L2 제곱 가중치 규제 / =lambda

prameters={'n_estimator':[100,200,300],
           'learnig_rate':[0.1,0.2,0.3,0.5],
           'max_depth':[None,2,3,4,5,6],
           'min_child_weight':[0.1,0.5,1,5],
           'reg_alpha':[0,0.1,1,10],
           'reg_lambda':[0,0.1,1,10]
           }



'''
prameters={'n_estimator':[100],
           'learnig_rate':[0.3],
           'max_depth':[6],
           'gamma':[0],
           'min_child_weight':[1],
           'subsample':[1],
           'colsample_bytree':[1],
           'colsample_bylevel':[1],
           'colsample_bynod':[1],
           'reg_alpha':[0],
           'reg_lambda':[1]
           }

# 2. 모델
xgb=XGBClassifier(random_state=123)
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
# 최상의 점수 :  0.9626373626373628

# 최상의 매개변수(파라미터) :  {'learnig_rate': 0.1, 'n_estimator': 100}
# 최상의 점수 :  0.9626373626373628

# 최상의 매개변수(파라미터) :  {'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 3, 'n_estimator': 100}
# 최상의 점수 :  0.9714285714285715

# 최상의 매개변수(파라미터) :  {'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0, 'n_estimator': 100}
# 최상의 점수 :  0.9714285714285715

# 최상의 매개변수(파라미터) :  {'colsample_bytree': 0.6, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0, 'n_estimator': 100, 'subsample': 0.6}
# 최상의 점수 :  0.9758241758241759

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 1, 'colsample_bytree': 0.6, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0, 'n_estimator': 100, 'subsample': 0.6}
# 최상의 점수 :  0.9758241758241759

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 1, 'colsample_bynod': 0, 'colsample_bytree': 0.6, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0, 'n_estimator': 100, 'subsample': 0.6}      
# 최상의 점수 :  0.9758241758241759

# 최상의 매개변수(파라미터) :  {'colsample_bylevel': 1, 'colsample_bynod': 0, 'colsample_bytree': 0.6, 'gamma': 0, 'learnig_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0, 'n_estimator': 100, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.6}
# 최상의 점수 :  0.9758241758241759
# model.score: 0.9912280701754386
'''