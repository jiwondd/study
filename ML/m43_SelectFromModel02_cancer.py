import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target
# print(x.shape) (569, 30)
x=np.delete(x,[0,1],axis=1)
# print(x.shape) (569, 28)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=123)

# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
model=XGBClassifier(random_state=100,
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
acc=accuracy_score(y_test, y_predict)
print('진짜 최종 test 점수 : ' , acc)
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
    
    selection_model=XGBClassifier(n_jobs=-1,
                                 random_state=100,
                                 n_estimators=100,
                                 learning_rate=0.3,
                                 max_depth=6,
                                 gamma=0)
    selection_model.fit(select_x_train,y_train)
    y_predict=selection_model.predict(select_x_test)
    score=accuracy_score(y_test,y_predict)
    print("Thresh=%.3f,n=%d, acc:%.2f%%"
          #소수점3개까지,정수,소수점2개까지
          %(thresh,select_x_train.shape[1],score*100))
    
'''
model.score: 0.9912280701754386
진짜 최종 test 점수 :  0.9912280701754386
걸린시간: 0.13
---------------------------------
[0.01911217 0.03170424 0.         0.00991825 0.01046152 0.00753364
 0.06238323 0.03759076 0.00059016 0.00264279 0.01457083 0.
 0.00083066 0.00769215 0.01308586 0.00765252 0.00316563 0.00229688
 0.00249091 0.00352028 0.01198188 0.01504461 0.29318932 0.32412094
 0.00909379 0.00494332 0.01575371 0.0782041  0.00290343 0.00752243]
---------------------------------
(455, 7) (114, 7)
Thresh=0.019,n=7, acc:95.61%
(455, 6) (114, 6)
Thresh=0.032,n=6, acc:94.74%
(455, 30) (114, 30)
Thresh=0.000,n=30, acc:99.12%
(455, 14) (114, 14)
Thresh=0.010,n=14, acc:99.12%
(455, 13) (114, 13)
Thresh=0.010,n=13, acc:99.12%
(455, 18) (114, 18)
Thresh=0.008,n=18, acc:99.12%
(455, 4) (114, 4)
Thresh=0.062,n=4, acc:95.61%
(455, 5) (114, 5)
Thresh=0.038,n=5, acc:94.74%
(455, 28) (114, 28)
Thresh=0.001,n=28, acc:99.12%
(455, 24) (114, 24)
Thresh=0.003,n=24, acc:100.00%
(455, 10) (114, 10)
Thresh=0.015,n=10, acc:97.37%
(455, 30) (114, 30)
Thresh=0.000,n=30, acc:99.12%
(455, 27) (114, 27)
Thresh=0.001,n=27, acc:99.12%
(455, 16) (114, 16)
Thresh=0.008,n=16, acc:99.12%
(455, 11) (114, 11)
Thresh=0.013,n=11, acc:97.37%
(455, 17) (114, 17)
Thresh=0.008,n=17, acc:98.25%
(455, 22) (114, 22)
Thresh=0.003,n=22, acc:99.12%
(455, 26) (114, 26)
Thresh=0.002,n=26, acc:100.00%
(455, 25) (114, 25)
Thresh=0.002,n=25, acc:100.00%
(455, 21) (114, 21)
Thresh=0.004,n=21, acc:100.00%
(455, 12) (114, 12)
Thresh=0.012,n=12, acc:96.49%
(455, 9) (114, 9)
Thresh=0.015,n=9, acc:96.49%
(455, 2) (114, 2)
Thresh=0.293,n=2, acc:89.47%
(455, 1) (114, 1)
Thresh=0.324,n=1, acc:85.09%
(455, 15) (114, 15)
Thresh=0.009,n=15, acc:99.12%
(455, 20) (114, 20)
Thresh=0.005,n=20, acc:99.12%
(455, 8) (114, 8)
Thresh=0.016,n=8, acc:95.61%
(455, 3) (114, 3)
Thresh=0.078,n=3, acc:94.74%
(455, 23) (114, 23)
Thresh=0.003,n=23, acc:97.37%
(455, 19) (114, 19)
Thresh=0.008,n=19, acc:100.00%


***************************************
model.score: 1.0
진짜 최종 test 점수 :  1.0
걸린시간: 0.09
---------------------------------
[0.         0.01996664 0.00373773 0.00491998 0.04807429 0.07985299
 0.00095339 0.00423668 0.03215441 0.         0.01705772 0.00968759
 0.01038092 0.00442247 0.01051008 0.02244038 0.00118794 0.00695529
 0.03496855 0.02375727 0.19791481 0.32363737 0.00792101 0.0085387
 0.01987661 0.08667586 0.00592384 0.01424738]
---------------------------------
(455, 28) (114, 28)
Thresh=0.000,n=28, acc:100.00%
(455, 10) (114, 10)
Thresh=0.020,n=10, acc:99.12%
(455, 24) (114, 24)
Thresh=0.004,n=24, acc:100.00%
(455, 21) (114, 21)
Thresh=0.005,n=21, acc:100.00%
(455, 5) (114, 5)
Thresh=0.048,n=5, acc:94.74%
(455, 4) (114, 4)
Thresh=0.080,n=4, acc:94.74%
(455, 26) (114, 26)
Thresh=0.001,n=26, acc:100.00%
(455, 23) (114, 23)
Thresh=0.004,n=23, acc:100.00%
(455, 7) (114, 7)
Thresh=0.032,n=7, acc:94.74%
(455, 28) (114, 28)
Thresh=0.000,n=28, acc:100.00%
(455, 12) (114, 12)
Thresh=0.017,n=12, acc:98.25%
(455, 16) (114, 16)
Thresh=0.010,n=16, acc:99.12%
(455, 15) (114, 15)
Thresh=0.010,n=15, acc:99.12%
(455, 22) (114, 22)
Thresh=0.004,n=22, acc:99.12%
(455, 14) (114, 14)
Thresh=0.011,n=14, acc:98.25%
(455, 9) (114, 9)
Thresh=0.022,n=9, acc:99.12%
(455, 25) (114, 25)
Thresh=0.001,n=25, acc:100.00%
(455, 19) (114, 19)
Thresh=0.007,n=19, acc:100.00%
(455, 6) (114, 6)
Thresh=0.035,n=6, acc:94.74%
(455, 8) (114, 8)
Thresh=0.024,n=8, acc:98.25%
(455, 2) (114, 2)
Thresh=0.198,n=2, acc:89.47%
(455, 1) (114, 1)
Thresh=0.324,n=1, acc:85.09%
(455, 18) (114, 18)
Thresh=0.008,n=18, acc:99.12%
(455, 17) (114, 17)
Thresh=0.009,n=17, acc:98.25%
(455, 11) (114, 11)
Thresh=0.020,n=11, acc:98.25%
(455, 3) (114, 3)
Thresh=0.087,n=3, acc:94.74%
(455, 20) (114, 20)
Thresh=0.006,n=20, acc:99.12%
(455, 13) (114, 13)
Thresh=0.014,n=13, acc:97.37%

'''