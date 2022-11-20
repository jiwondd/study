import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from icecream import ic

# 1. 데이터
datasets = load_breast_cancer()

df=pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df.head())

x_train, x_test, y_train, y_test= train_test_split(
    datasets.data,datasets.target, train_size=0.8, random_state=123, stratify=datasets.target
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_train.shape,y_train.shape) #(455, 30) (455,)


# 2. 모델
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
lr=LogisticRegression()
knn=KNeighborsClassifier(n_neighbors=8)

xg=XGBClassifier()
lg=LGBMClassifier()
cat=CatBoostClassifier(verbose=0)

model=VotingClassifier(
    estimators=[('XG',xg),('LG',lg),('cat',cat)],
    voting='soft' # hard
)

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가, 예측
y_pred=model.predict(x_test)
score=accuracy_score(y_test,y_pred)
print('보팅결과 :' ,round(score,4) ) 
print('time :',np.round(end-start,2))

classifiers=[xg,lg,cat]
for model2 in classifiers:
    model2.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    score2=accuracy_score(y_test,y_predict)
    class_name=model2.__class__.__name__
    print('{0}정확도:{1:4f}'.format(class_name,score2))

  
# 보팅결과 : 0.9825
# time : 2.57
# XGBClassifier정확도:0.982456
# LGBMClassifier정확도:0.982456
# CatBoostClassifier정확도:0.982456
