# 피쳐하나 삭제해서 비교해보기

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target
# print(x.shape) #(442, 10)
x = np.delete(x,1,axis=1)
# print(x.shape) (442, 9)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import time

# lr=LogisticRegression()
knn=KNeighborsRegressor(n_neighbors=8)
xg=XGBRegressor()
lg=LGBMRegressor()
cat=CatBoostRegressor(verbose=0)

model=VotingRegressor(
    estimators=[('XG',xg),('LG',lg),('cat',cat)]
)

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가, 예측
y_pred=model.predict(x_test)
score=r2_score(y_test,y_pred)
print('보팅결과 :' ,round(score,4) ) 
print('time :',np.round(end-start,2))

Regressor=[xg,lg,cat]
for model2 in Regressor:
    model2.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    score2=r2_score(y_test,y_predict)
    class_name=model2.__class__.__name__
    print('{0}정확도:{1:4f}'.format(class_name,score2))
    
# 보팅결과 : 0.4754
# time : 1.0
# XGBRegressor정확도:0.475376
# LGBMRegressor정확도:0.475376
# CatBoostRegressor정확도:0.475376
