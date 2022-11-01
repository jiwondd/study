import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target

x = np.delete(x,[1,3,5],axis=1)
# print(x.shape) (178, 13)->(178, 10)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=72)

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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV #이름은 리그레션인데 "이진"분류에 쓴다 (시그모이드로 결과를 뺀다)

model=BaggingClassifier(RandomForestClassifier(),
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=72
                        )

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가, 예측
print(model.score(x_test,y_test)) 
print('time :',np.round(end-start,2))

# 0.9444444444444444 <-LogisticRegression / time : 1.07
# 0.9166666666666666 <-GradientBoostingClassifier / time : 3.72
# 0.9166666666666666 <-DecisionTreeClassifier / time : 1.06
# 0.9722222222222222 <-RandomForestClassifier / time : 2.64

