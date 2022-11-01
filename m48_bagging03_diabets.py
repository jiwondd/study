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

# 0.5223651147193191 <-DecisionTreeRegressor / time : 1.05
# 0.5458034358177335 <-RandomForestRegressor / time : 3.44
# 0.5708844672066027 <-GradientBoostingRegressor / time : 1.79