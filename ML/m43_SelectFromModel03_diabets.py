import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(x.shape) #(442, 10)
x=np.delete(x,[0],axis=1)
print(x.shape) #(442, 8)

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
          eval_metric='rmse'
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
model.score: 0.48490594952801225
진짜 최종 test 점수 :  0.48490594952801225
걸린시간: 0.09
---------------------------------
[0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
 0.06012432 0.09595273 0.30483875 0.06629313]
---------------------------------
(353, 10) (89, 10)
Thresh=0.032,n=10, r2:46.02%
(353, 9) (89, 9)
Thresh=0.045,n=9, r2:45.95%
(353, 2) (89, 2)
Thresh=0.218,n=2, r2:34.70%
(353, 4) (89, 4)
Thresh=0.082,n=4, r2:36.30%
(353, 8) (89, 8)
Thresh=0.047,n=8, r2:36.12%
(353, 7) (89, 7)
Thresh=0.048,n=7, r2:38.43%
(353, 6) (89, 6)
Thresh=0.060,n=6, r2:44.11%
(353, 3) (89, 3)
Thresh=0.096,n=3, r2:37.55%
(353, 1) (89, 1)
Thresh=0.305,n=1, r2:-5.81%
(353, 5) (89, 5)
Thresh=0.066,n=5, r2:42.75%

model.score: 0.4879904227399403
진짜 최종 test 점수 :  0.4879904227399403
걸린시간: 0.09
---------------------------------
[0.03015558 0.19155757 0.08232232 0.05387235 0.05853077 0.08402364
 0.10219601 0.30457193 0.09276979]
---------------------------------
(353, 9) (89, 9)
Thresh=0.030,n=9, r2:45.95%
(353, 2) (89, 2)
Thresh=0.192,n=2, r2:34.70%
(353, 6) (89, 6)
Thresh=0.082,n=6, r2:44.11%
(353, 8) (89, 8)
Thresh=0.054,n=8, r2:36.12%
(353, 7) (89, 7)
Thresh=0.059,n=7, r2:38.43%
(353, 5) (89, 5)
Thresh=0.084,n=5, r2:38.69%
(353, 3) (89, 3)
Thresh=0.102,n=3, r2:37.55%
(353, 1) (89, 1)
Thresh=0.305,n=1, r2:-5.81%
(353, 4) (89, 4)
Thresh=0.093,n=4, r2:33.58%
'''