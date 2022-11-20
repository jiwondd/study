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
        train_size=0.7,shuffle=True, random_state=1234)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
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

model=BaggingRegressor(XGBRegressor(),
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=1234
                        )

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가, 예측
print(model.score(x_test,y_test)) 
print('time :',np.round(end-start,2))

# 0.8612672046310272 <- DecisionTreeRegressor / time : 1.09
# 0.8376645688564723 <- RandomForestRegressor / time : 3.52
# 0.8686788455192399 <- GradientBoostingRegressor / time : 1.92
# 0.8962654131536463 <- XGBRegressor / time : 3.22
