import numpy as np
import pandas as pd
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
warnings.filterwarnings(action='ignore')


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

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,
                                                  random_state=123,shuffle=True)

kFold=KFold(n_splits=5, shuffle=True,random_state=123)

parameters={'n_estimator':[100,200,300],
           'learnig_rate':[0.1,0.2,0.3,0.5],
           'max_depth':[None,2,3,4,5,6],
           'min_child_weight':[0.1,0.5,1,5],
           'reg_alpha':[0,0.1,1,10],
           'reg_lambda':[0,0.1,1,10]
           }

# 2. 모델
xgb=XGBRegressor(random_state=123)
model=GridSearchCV(xgb,parameters,cv=kFold,n_jobs=8)

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
best_params=model.best_params_
print('최상의 매개변수(파라미터) : ', best_params )
print('최상의 점수 : ',model.best_score_)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('걸린시간 :',np.round(end-start,2))


