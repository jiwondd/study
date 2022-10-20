import numpy as np
import pandas as pd
from csv import reader
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
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!

# print(train_set.isnull().sum())
# print(train_set.info())
# print(train_set.shape) #(1459, 10)
# print(test_set.shape) #(715, 9)

train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
y=train_set['count']
print(x.shape) #(1459, 9)
print(y.shape) #(1459, 9)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

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

# 결과: 0.854683534348966
