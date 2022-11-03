import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count','hour_bef_precipitation','hour_bef_humidity'],axis=1)
y=train_set['count']

# print(x.shape) #(1328, 9)->(1328, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
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

# 0.88044022413361 <- XGBRegressor / time : 3.75
# 0.8533510701294347 <- GradientBoostingClassifier / time : 2.4 
# 0.9237357643935747 <- DecisionTreeRegressor / time : 1.08
# 0.909196889471297 <- RandomForestRegressor / time : 6.44

# 보팅결과 : 0.8595
# time : 1.15
# XGBRegressor정확도:0.859470
# LGBMRegressor정확도:0.859470
# CatBoostRegressor정확도:0.859470
