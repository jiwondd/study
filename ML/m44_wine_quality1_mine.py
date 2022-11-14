import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from cProfile import label
from sklearn.ensemble import RandomForestClassifier

#.1 데이터
path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=0, sep=';')
# print(data_set.shape) (4898, 11)                             ㄴ기준으로 컬럼을 나눠줘

#.1 데이터
y=data_set['quality']
x=data_set.drop(['quality'],axis=1)

le=LabelEncoder()
y=le.fit_transform(y)

scaler=MinMaxScaler()
scaler.fit(x)
data_set=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,
                                                  random_state=123,shuffle=True)

kFold=StratifiedKFold(n_splits=2, shuffle=True,random_state=123)

XG_parameters={'n_estimators':[100],
            'learning_rate':[0.001],
            'max_depth':[3],
            'gamma':[0],
            'min_child_weight':[1],
            'subsample':[0.1],
            'colsample_bytree':[0],
            'colsample_bylevel':[0],
            'colsample_bynode':[0],
            'reg_alpha':[2],
            'reg_lambda':[2]
           }

RF_parameters = [
    {'n_estimators':[100,200,300],'max_depth':[1,3,6,8,10],'min_samples_leaf':[4,8,12]}
    # {'max_depth':[1,3,6,8,10]},
    # {'min_samples_leaf':[4,8,12]},
    # {'min_samples_split':[2,4,6,8]},
    # {'max_depth' : [6, 8, 10, 12]}
]

# 2. 모델구성
xg =XGBClassifier(random_state=123)
rf = RandomForestClassifier(random_state=123)
model=GridSearchCV(rf,RF_parameters,cv=kFold,n_jobs=8)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 
best_params=model.best_params_
print('best_params : ', best_params )

# (랜덤스테이트 42,1004,777, 1234 다 별로)
# model.score: 0.47551020408163264 <-xg
# model.score: 0.6908163265306122 <-rf

# 랜덤스테이트로 결과값이 많이 달라지다는건 데이터의 분포가 골고루 되어있지 않다는 말이다
# 분류모델을 유니크로 라벨을 보고 해야한다. 분포가 어떤지 어뜨캐 생겨먹었는지
