import numpy as np
import pandas as pd
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
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
    
'''
model.score: -0.5697346452196475
진짜 최종 test 점수 :  -0.5697346452196475
걸린시간: 0.15
---------------------------------
[0.3498928  0.1051626  0.37146357 0.02119369 0.02538931 0.02375728
 0.04428132 0.03122395 0.02763546]
---------------------------------
(1062, 2) (266, 2)
Thresh=0.350,n=2, r2:66.00%
(1062, 3) (266, 3)
Thresh=0.105,n=3, r2:68.03%
(1062, 1) (266, 1)
Thresh=0.371,n=1, r2:2.79%
(1062, 9) (266, 9)
Thresh=0.021,n=9, r2:77.22%
(1062, 7) (266, 7)
Thresh=0.025,n=7, r2:77.13%
(1062, 8) (266, 8)
Thresh=0.024,n=8, r2:77.98%
(1062, 4) (266, 4)
Thresh=0.044,n=4, r2:69.54%
(1062, 5) (266, 5)
Thresh=0.031,n=5, r2:73.86%
(1062, 6) (266, 6)
Thresh=0.028,n=6, r2:78.76%
'''
