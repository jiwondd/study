from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

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

# 0.8057737565738272 <-DecisionTreeRegressor / time : 2.38
# 0.8003920770410915 <-RandomForestRegressor / time : 91.15
# 0.7783150887379254 <-GradientBoostingRegressor / time : 26.1
# 0.8474068641787094 <-XGBRegressor / time : 31.09

# 보팅결과 : 0.8479
# time : 2.57
# XGBRegressor정확도:0.847921
# LGBMRegressor정확도:0.847921
# CatBoostRegressor정확도:0.847921