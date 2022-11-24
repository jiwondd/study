import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target

# x = np.delete(x,[1,8,11],axis=1)
# print(x.shape) #(506, 13)->(506, 10)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

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

# 0.8612672046310272 <- DecisionTreeRegressor / time : 1.09
# 0.8376645688564723 <- RandomForestRegressor / time : 3.52
# 0.8686788455192399 <- GradientBoostingRegressor / time : 1.92
# 0.8604801257327843 <- XGBRegressor / time : 3.22

# 보팅결과 : 0.8811
# time : 1.14
# XGBRegressor정확도:0.881061
# LGBMRegressor정확도:0.881061
# CatBoostRegressor정확도:0.881061
