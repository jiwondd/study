import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier, XGBRegressor

#.1 데이터
path='./_data/kaggle_bike/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!

# print(train_set.shape) #(10886, 12)
# print(test_set.shape) #(6493, 9)

train_set['datetime']=pd.to_datetime(train_set['datetime'])
train_set['year']=train_set['datetime'].dt.year
train_set['month']=train_set['datetime'].dt.month
train_set['day']=train_set['datetime'].dt.day
train_set['hour']=train_set['datetime'].dt.hour
train_set.drop(['datetime','day','year'],inplace=True,axis=1)

train_set['month']=train_set['month'].astype('category')
train_set['hour']=train_set['hour'].astype('category')

train_set=pd.get_dummies(train_set,columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp',inplace=True,axis=1)

test_set['datetime']=pd.to_datetime(test_set['datetime'])
test_set['month']=test_set['datetime'].dt.month
test_set['hour']=test_set['datetime'].dt.hour
test_set['month']=test_set['month'].astype('category')
test_set['hour']=test_set['hour'].astype('category')

test_set=pd.get_dummies(test_set,columns=['season','weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count','humidity'], axis=1)
y =train_set['count']

outliers=EllipticEnvelope(contamination=.2) 
outliers.fit(x)
outliers1=outliers.predict(x)
print(outliers1)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=123)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
model=XGBRegressor(random_state=100,
                    n_estimator=100,
                    learnig_rate=0.3,
                    max_depth=6,
                    gamma=0)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train,early_stopping_rounds=100,
          eval_set=[(x_test,y_test)],
          eval_metric='error'
          )
end=time.time()

# 4. 평가
result=model.score(x_test,y_test)
print('model.score:',result) 
y_predict=model.predict(x_test)
r2=r2_score(y_test, y_predict)
print('진짜 최종 test 점수 : ' , r2)
print('걸린시간:',np.round(end-start,2))
print('---------------------------------')
print(model.feature_importances_)
thresholds=model.feature_importances_
print('---------------------------------')

for thresh in thresholds :
    selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train=selection.transform(x_train)
    select_x_test=selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)
    
    selection_model=XGBRegressor(n_jobs=-1,
                                 random_state=100,
                                 n_estimators=100,
                                 learning_rate=0.3,
                                 max_depth=6,
                                 gamma=0)
    selection_model.fit(select_x_train,y_train)
    y_predict=selection_model.predict(select_x_test)
    score=r2_score(y_test,y_predict)
    print("Thresh=%.3f,n=%d, r2:%.2f%%"
          #소수점3개까지,정수,소수점2개까지
          %(thresh,select_x_train.shape[1],score*100))

# model.score: -0.1592320749664251
# 진짜 최종 test 점수 :  -0.1592320749664251
# 걸린시간: 0.34
# ---------------------------------
# [0.02516489 0.20438866 0.07404342 0.0120362  0.04607036 0.4032188
#  0.         0.02664553 0.01880466 0.         0.03248577 0.00971194
#  0.1474298  0.        ]
# ---------------------------------
# (10777, 8) (109, 8)
# Thresh=0.025,n=8, r2:88.35%
# (10777, 2) (109, 2)
# Thresh=0.204,n=2, r2:60.31%
# (10777, 4) (109, 4)
# Thresh=0.074,n=4, r2:76.55%
# (10777, 10) (109, 10)
# Thresh=0.012,n=10, r2:87.29%
# (10777, 5) (109, 5)
# Thresh=0.046,n=5, r2:89.03%
# (10777, 1) (109, 1)
# Thresh=0.403,n=1, r2:47.99%
# (10777, 14) (109, 14)
# Thresh=0.000,n=14, r2:87.92%
# (10777, 7) (109, 7)
# Thresh=0.027,n=7, r2:87.15%
# (10777, 9) (109, 9)
# Thresh=0.019,n=9, r2:86.95%
# (10777, 14) (109, 14)
# Thresh=0.000,n=14, r2:87.92%
# (10777, 6) (109, 6)
# Thresh=0.032,n=6, r2:87.01%
# (10777, 11) (109, 11)
# Thresh=0.010,n=11, r2:87.92%
# (10777, 3) (109, 3)
# Thresh=0.147,n=3, r2:63.36%
# (10777, 14) (109, 14)
# Thresh=0.000,n=14, r2:87.92%
