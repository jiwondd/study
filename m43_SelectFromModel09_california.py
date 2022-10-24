import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=fetch_california_housing()
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
model.score: 0.05566499554272375
진짜 최종 test 점수 :  0.05566499554272375
걸린시간: 0.73
---------------------------------
[0.4407601  0.07022503 0.04738959 0.02539877 0.02791627 0.15494774
 0.10437356 0.12898892]
---------------------------------
(16512, 1) (4128, 1)
Thresh=0.441,n=1, r2:46.64%
(16512, 5) (4128, 5)
Thresh=0.070,n=5, r2:83.90%
(16512, 6) (4128, 6)
Thresh=0.047,n=6, r2:83.75%
(16512, 8) (4128, 8)
Thresh=0.025,n=8, r2:83.32%
(16512, 7) (4128, 7)
Thresh=0.028,n=7, r2:83.99%
(16512, 2) (4128, 2)
Thresh=0.155,n=2, r2:58.05%
(16512, 4) (4128, 4)
Thresh=0.104,n=4, r2:83.55%
(16512, 3) (4128, 3)
Thresh=0.129,n=3, r2:72.42%
'''