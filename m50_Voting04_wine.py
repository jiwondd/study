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
    

# 0.9444444444444444 <-LogisticRegression / time : 1.07
# 0.9166666666666666 <-GradientBoostingClassifier / time : 3.72
# 0.9166666666666666 <-DecisionTreeClassifier / time : 1.06
# 0.9722222222222222 <-RandomForestClassifier / time : 2.64

# 보팅결과 : 0.9722
# time : 1.31
# XGBClassifier정확도:0.972222
# LGBMClassifier정확도:0.972222
# CatBoostClassifier정확도:0.972222