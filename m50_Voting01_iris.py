# 피쳐 임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성 후
# 모델별로 돌려서 결과도출/ 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets=load_iris()
x=datasets.data
y=datasets.target

print(x.shape,y.shape) #(455, 30) (455,)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_train.shape,y_train.shape) #(120, 4) (120,)

# 2. 모델
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import time

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


# 보팅결과 : 0.9667
# time : 0.8
# XGBClassifier정확도:0.966667
# LGBMClassifier정확도:0.966667
# CatBoostClassifier정확도:0.966667