import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
warnings.filterwarnings(action='ignore')

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,
                                                  random_state=123,shuffle=True)

kFold=KFold(n_splits=5, shuffle=True,random_state=123)

parameters={'n_estimator':[100,200,300],
           'learnig_rate':[0.1,0.2,0.3,0.5],
           'max_depth':[None,2,3,4,5,6],
           'min_child_weight':[0.1,0.5,1,5],
           'reg_alpha':[0,0.1,1,10],
           'reg_lambda':[0,0.1,1,10]
           }

# 2. 모델
xgb=XGBRegressor(random_state=123)
model=GridSearchCV(xgb,parameters,cv=kFold,n_jobs=8)

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
best_params=model.best_params_
print('최상의 매개변수(파라미터) : ', best_params )
print('최상의 점수 : ',model.best_score_)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('걸린시간 :',np.round(end-start,2))
