import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=fetch_covtype()
x=datasets.data
y=datasets.target

le=LabelEncoder()
y=le.fit_transform(y)

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
model.score: 0.9666666666666667
진짜 최종 test 점수 :  0.9666666666666667
걸린시간: 0.12
---------------------------------
[0.02430454 0.02472077 0.7376847  0.21328998]
---------------------------------
(120, 4) (30, 4)
Thresh=0.024,n=4, acc:93.33%
(120, 3) (30, 3)
Thresh=0.025,n=3, acc:93.33%
(120, 1) (30, 1)
Thresh=0.738,n=1, acc:90.00%
(120, 2) (30, 2)
Thresh=0.213,n=2, acc:96.67%
'''
