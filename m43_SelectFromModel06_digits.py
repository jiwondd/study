import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=load_digits()
x=datasets.data
y=datasets.target

print(y.shape)

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
          eval_metric='merror'
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
model.score: 0.975
진짜 최종 test 점수 :  0.975
걸린시간: 0.49
---------------------------------
[0.         0.06415373 0.01059751 0.00873502 0.00434788 0.03380749        
 0.00921624 0.01642124 0.         0.01852775 0.01413111 0.01034479        
 0.00841835 0.01131467 0.00558309 0.00333757 0.         0.00507369        
 0.00538672 0.04030624 0.01079499 0.04513224 0.00413447 0.
 0.         0.00495149 0.02641704 0.00841728 0.02751666 0.01865147        
 0.01018624 0.         0.         0.07354812 0.00440697 0.00674871        
 0.04887262 0.01661008 0.02459871 0.         0.         0.00697014        
 0.03422127 0.04050089 0.0116045  0.01881086 0.0242893  0.
 0.         0.00862246 0.005208   0.00565973 0.01122131 0.0095808
 0.02874239 0.         0.         0.04148491 0.00863673 0.00476112        
 0.06507792 0.01276453 0.04295828 0.01819468]
---------------------------------
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 3) (360, 3)
Thresh=0.064,n=3, acc:39.72%
(1437, 29) (360, 29)
Thresh=0.011,n=29, acc:96.11%
(1437, 34) (360, 34)
Thresh=0.009,n=34, acc:95.56%
(1437, 49) (360, 49)
Thresh=0.004,n=49, acc:97.78%
(1437, 11) (360, 11)
Thresh=0.034,n=11, acc:87.50%
(1437, 33) (360, 33)
Thresh=0.009,n=33, acc:96.11%
(1437, 22) (360, 22)
Thresh=0.016,n=22, acc:95.83%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 19) (360, 19)
Thresh=0.019,n=19, acc:96.39%
(1437, 23) (360, 23)
Thresh=0.014,n=23, acc:96.39%
(1437, 30) (360, 30)
Thresh=0.010,n=30, acc:95.83%
(1437, 37) (360, 37)
Thresh=0.008,n=37, acc:96.39%
(1437, 26) (360, 26)
Thresh=0.011,n=26, acc:95.56%
(1437, 42) (360, 42)
Thresh=0.006,n=42, acc:96.39%
(1437, 51) (360, 51)
Thresh=0.003,n=51, acc:97.50%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 45) (360, 45)
Thresh=0.005,n=45, acc:97.22%
(1437, 43) (360, 43)
Thresh=0.005,n=43, acc:96.94%
(1437, 9) (360, 9)
Thresh=0.040,n=9, acc:81.67%
(1437, 28) (360, 28)
Thresh=0.011,n=28, acc:95.83%
(1437, 5) (360, 5)
Thresh=0.045,n=5, acc:60.28%
(1437, 50) (360, 50)
Thresh=0.004,n=50, acc:97.22%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 46) (360, 46)
Thresh=0.005,n=46, acc:97.50%
(1437, 14) (360, 14)
Thresh=0.026,n=14, acc:94.72%
(1437, 38) (360, 38)
Thresh=0.008,n=38, acc:96.11%
(1437, 13) (360, 13)
Thresh=0.028,n=13, acc:90.56%
(1437, 18) (360, 18)
Thresh=0.019,n=18, acc:95.28%
(1437, 31) (360, 31)
Thresh=0.010,n=31, acc:95.83%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 1) (360, 1)
Thresh=0.074,n=1, acc:28.89%
(1437, 48) (360, 48)
Thresh=0.004,n=48, acc:97.22%
(1437, 40) (360, 40)
Thresh=0.007,n=40, acc:96.67%
(1437, 4) (360, 4)
Thresh=0.049,n=4, acc:51.67%
(1437, 21) (360, 21)
Thresh=0.017,n=21, acc:96.39%
Thresh=0.025,n=15, acc:94.72%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 39) (360, 39)
Thresh=0.007,n=39, acc:96.39%
(1437, 10) (360, 10)
Thresh=0.034,n=10, acc:86.11%
(1437, 8) (360, 8)
Thresh=0.041,n=8, acc:77.78%
(1437, 25) (360, 25)
Thresh=0.012,n=25, acc:95.83%
(1437, 17) (360, 17)
Thresh=0.019,n=17, acc:95.56%
(1437, 16) (360, 16)
Thresh=0.024,n=16, acc:94.44%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 36) (360, 36)
Thresh=0.009,n=36, acc:95.56%
(1437, 44) (360, 44)
Thresh=0.005,n=44, acc:96.94%
(1437, 41) (360, 41)
Thresh=0.006,n=41, acc:96.67%
(1437, 27) (360, 27)
Thresh=0.011,n=27, acc:95.28%
(1437, 32) (360, 32)
Thresh=0.010,n=32, acc:95.83%
(1437, 12) (360, 12)
Thresh=0.029,n=12, acc:90.28%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 64) (360, 64)
Thresh=0.000,n=64, acc:97.50%
(1437, 7) (360, 7)
Thresh=0.041,n=7, acc:68.33%
(1437, 35) (360, 35)
Thresh=0.009,n=35, acc:95.83%
(1437, 47) (360, 47)
Thresh=0.005,n=47, acc:97.50%
(1437, 2) (360, 2)
Thresh=0.065,n=2, acc:38.33%
(1437, 24) (360, 24)
Thresh=0.013,n=24, acc:96.11%
(1437, 6) (360, 6)
Thresh=0.043,n=6, acc:68.61%
(1437, 20) (360, 20)
Thresh=0.018,n=20, acc:95.83%
'''