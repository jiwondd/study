import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=load_wine()
x=datasets.data
y=datasets.target

# print(x.shape) #(442, 10)
x=np.delete(x,[0],axis=1)
# print(x.shape) #(442, 8)

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
model.score: 0.9444444444444444
진짜 최종 test 점수 :  0.9444444444444444
걸린시간: 0.08
---------------------------------
[0.06331411 0.08874952 0.01123517 0.00261207 0.03230456 0.02632481
 0.22729877 0.         0.01110675 0.28431928 0.03778306 0.02962446
 0.18532743]
---------------------------------
(142, 5) (36, 5)
Thresh=0.063,n=5, acc:94.44%
(142, 4) (36, 4)
Thresh=0.089,n=4, acc:94.44%
(142, 10) (36, 10)
Thresh=0.011,n=10, acc:91.67%
(142, 12) (36, 12)
Thresh=0.003,n=12, acc:91.67%
(142, 7) (36, 7)
Thresh=0.032,n=7, acc:91.67%
(142, 9) (36, 9)
Thresh=0.026,n=9, acc:94.44%
(142, 2) (36, 2)
Thresh=0.227,n=2, acc:91.67%
(142, 13) (36, 13)
Thresh=0.000,n=13, acc:91.67%
(142, 11) (36, 11)
Thresh=0.011,n=11, acc:91.67%
(142, 1) (36, 1)
Thresh=0.284,n=1, acc:63.89%
(142, 6) (36, 6)
Thresh=0.038,n=6, acc:94.44%
(142, 8) (36, 8)
Thresh=0.030,n=8, acc:91.67%
(142, 3) (36, 3)
Thresh=0.185,n=3, acc:91.67%

'''