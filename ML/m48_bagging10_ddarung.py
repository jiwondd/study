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
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV #이름은 리그레션인데 "이진"분류에 쓴다 (시그모이드로 결과를 뺀다)

model=BaggingRegressor(GradientBoostingRegressor(),
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=123
                        )

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가, 예측
print(model.score(x_test,y_test)) 
print('time :',np.round(end-start,2))

# 0.88044022413361 <- XGBRegressor / time : 3.75
# 0.8533510701294347 <- GradientBoostingClassifier / time : 2.4 
# 0.9237357643935747 <- DecisionTreeRegressor / time : 1.08
# 0.909196889471297 <- RandomForestRegressor / time : 6.44

