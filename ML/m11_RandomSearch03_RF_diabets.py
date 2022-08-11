from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=777)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

parameters=[
    {'n_estimators':[100,200],'max_depth':[6,8,10,23]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],
     'n_jobs':[-1,2,4]},
]

#2. 모델구성
model=RandomizedSearchCV(RandomForestRegressor(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)


# 3. 컴파일 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

print("최적의 매개변수: ",model.best_estimator_)
print("최적의 파라미터: ",model.best_params_)
print("best_score: ",model.best_score_)
print("model.score:",model.score(x_test,y_test))

#4. 평가, 예측
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 score :', r2)
y_pred_best=model.best_estimator_.predict(x_test)
print("최적 튠 r2 : ",r2_score(y_test,y_pred_best))
print('걸린시간:',np.round(end-start,2))

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, n_jobs=4)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 2, 'n_jobs': 4}
# best_score:  0.46146949009288657
# model.score: 0.38710381659703685
# r2 score : 0.38710381659703685  
# 최적 튠 r2 :  0.38710381659703696
# 걸린시간: 6.94

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=10, n_jobs=4)
# 최적의 파라미터:  {'n_jobs': 4, 'min_samples_split': 10, 'min_samples_leaf': 3}
# best_score:  0.4549962129337361
# model.score: 0.40742604815261896
# r2 score : 0.40742604815261907
# 최적 튠 r2 :  0.40742604815261907
# 걸린시간: 2.57