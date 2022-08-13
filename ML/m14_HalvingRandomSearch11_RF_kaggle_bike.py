#https://www.kaggle.com/competitions/bike-sharing-demand/submit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#.1 데이터
path='./_data/kaggle_bike/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!


#데이트 타임 연/월/일/시 로 컬럼 나누기
train_set['datetime']=pd.to_datetime(train_set['datetime']) #date time 열을 date time 속성으로 변경
#세부 날짜별 정보를 보기 위해 날짜 데이터를 년도, 월, 일, 시간으로 나눠준다.(분,초는 모든값이 0 이므로 추가하지않는다.)
train_set['year']=train_set['datetime'].dt.year
train_set['month']=train_set['datetime'].dt.month
train_set['day']=train_set['datetime'].dt.day
train_set['hour']=train_set['datetime'].dt.hour

#날짜와 시간에 관련된 피쳐에는 datetime, holiday, workingday,year,month,day,hour 이 있다.
#숫자형으로 나오는 holiday,workingday,month,hour만 쓰고 나머지 제거한다.

train_set.drop(['datetime','day','year'],inplace=True,axis=1) #datetime, day, year 제거하기

#month, hour은 범주형으로 변경해주기
train_set['month']=train_set['month'].astype('category')
train_set['hour']=train_set['hour'].astype('category')

#season과 weather은 범주형 피쳐이다. 두 피쳐 모두 숫자로 표현되어 있으니 문자로 변환해준다.
train_set=pd.get_dummies(train_set,columns=['season','weather'])

#casual과 registered는 test데이터에 존재하지 않기에 삭제한다.
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
#temp와 atemp는 상관관계가 아주 높고 두 피쳐의 의미가 비슷하기 때문에 temp만 사용한다.
train_set.drop('atemp',inplace=True,axis=1) #atemp 지우기

#위처럼 test_set도 적용하기
test_set['datetime']=pd.to_datetime(test_set['datetime'])

test_set['month']=test_set['datetime'].dt.month
test_set['hour']=test_set['datetime'].dt.hour

test_set['month']=test_set['month'].astype('category')
test_set['hour']=test_set['hour'].astype('category')

test_set=pd.get_dummies(test_set,columns=['season','weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y=train_set['count']

# x_train,x_test,y_train,y_test=train_test_split(x,y,
#         test_size=0.1,shuffle=True, random_state=777)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=777)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)


parameters=[
    {'n_estimators':[100,200],'max_depth':[6,8,10,23]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],
     'n_jobs':[-1,2,4]},
]

#2. 모델구성
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model=HalvingRandomSearchCV(RandomForestRegressor(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)


# 3. 컴파일 훈련
import time
start=time.time()
model.fit(x,y)
end=time.time()

print("최적의 매개변수: ",model.best_estimator_)
print("최적의 파라미터: ",model.best_params_)
print("best_score: ",model.best_score_)
print("model.score:",model.score(x,y))

#4. 평가, 예측
y_predict=model.predict(x)
r2=r2_score(y,y_predict)
print('r2 score :', r2)
y_pred_best=model.best_estimator_.predict(x)
print("최적 튠 r2 : ",r2_score(y,y_pred_best))
print('걸린시간:',np.round(end-start,2))

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=23, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 23, 'n_estimators': 200}
# best_score:  0.8574222124116323
# model.score: 0.8411523054910874
# r2 score : 0.8411523054910874
# 최적 튠 r2 :  0.8411523054910874
# 걸린시간: 57.49

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=3, n_jobs=4)
# 최적의 파라미터:  {'n_jobs': 4, 'min_samples_split': 3, 'min_samples_leaf': 3}
# best_score:  0.8539462024631381
# model.score: 0.8318147083275084
# r2 score : 0.8318147083275084
# 최적 튠 r2 :  0.8318147083275084
# 걸린시간: 10.99

# HalvingGridSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 399
# max_resources_: 10777
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 399
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 1197
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 3591
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 10773
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=23, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 23, 'n_estimators': 200}
# best_score:  0.8572421909560186
# model.score: 0.8441097289136442
# r2 score : 0.8441097289136442
# 최적 튠 r2 :  0.8441097289136442
# 걸린시간: 25.59

# HalvingRandomSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 7
# min_resources_: 10
# max_resources_: 10886
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 10
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# C:\Anaconda3\envs\tf282gpu\lib\site-packages\sklearn\metrics\_regression.py:682: UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
#   warnings.warn(msg, UndefinedMetricWarning)
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 30
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 90
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 270
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, n_jobs=4)
# 최적의 파라미터:  {'n_jobs': 4, 'min_samples_split': 2, 'min_samples_leaf': 3}
# best_score:  0.5713946792402267
# model.score: 0.9424854802980289
# r2 score : 0.9424854802980289
# 최적 튠 r2 :  0.9424854802980289
# 걸린시간: 9.26