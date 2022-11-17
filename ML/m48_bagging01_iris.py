# 피쳐 임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성 후
# 모델별로 돌려서 결과도출/ 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets=load_iris()
x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8, shuffle=True, random_state=1234)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV #이름은 리그레션인데 "이진"분류에 쓴다 (시그모이드로 결과를 뺀다)

model=BaggingClassifier(DecisionTreeClassifier(),
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
# 1.0 <-DecisionTreeClassifier / time : 1.0
# 1.0 <-RandomForestClassifier / time : 2.58
