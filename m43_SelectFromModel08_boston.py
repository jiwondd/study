import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=load_boston()
x=datasets.data
y=datasets.target

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
model.score: -2.7664561872790183
진짜 최종 test 점수 :  -2.7664561872790183걸린시간: 0.09
---------------------------------
[1.2563994e-02 1.1845288e-03 1.1291430e-02 1.7764691e-04 4.2042546e-02
 3.7618080e-01 1.6225524e-02 3.2118235e-02 9.3574487e-03 3.7309986e-02
 4.4186968e-02 9.4257453e-03 4.0793514e-01]
---------------------------------
(404, 8) (102, 8)
Thresh=0.013,n=8, r2:80.88%
(404, 12) (102, 12)
Thresh=0.001,n=12, r2:82.15%
(404, 9) (102, 9)
Thresh=0.011,n=9, r2:82.75%
(404, 13) (102, 13)
Thresh=0.000,n=13, r2:81.96%
(404, 4) (102, 4)
Thresh=0.042,n=4, r2:75.19%
(404, 2) (102, 2)
Thresh=0.376,n=2, r2:53.29%
(404, 7) (102, 7)
Thresh=0.016,n=7, r2:77.66%
(404, 6) (102, 6)
Thresh=0.032,n=6, r2:79.59%
(404, 11) (102, 11)
Thresh=0.009,n=11, r2:82.02%
(404, 5) (102, 5)
Thresh=0.037,n=5, r2:73.29%
(404, 3) (102, 3)
Thresh=0.044,n=3, r2:67.77%
(404, 10) (102, 10)
Thresh=0.009,n=10, r2:83.65%
(404, 1) (102, 1)
Thresh=0.408,n=1, r2:42.82%
'''