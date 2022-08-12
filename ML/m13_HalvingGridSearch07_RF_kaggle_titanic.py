import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
path='./_data/kaggle_titanic/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv')
train = train_set.drop(['PassengerId','Name', 'Ticket','Cabin'], axis = 1 )
test = test_set.drop(['Name', 'Ticket','Cabin'], axis= 1)

sex_train_dummies = pd.get_dummies(train['Sex'])
sex_test_dummies = pd.get_dummies(test['Sex'])

sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']

train.drop(['Sex'], axis=1, inplace=True)
test.drop(['Sex'], axis=1, inplace=True)

train = train.join(sex_train_dummies)
test = test.join(sex_test_dummies)

train["Age"].fillna(train["Age"].mean() , inplace=True)
test["Age"].fillna(train["Age"].mean() , inplace=True)

train["Embarked"].fillna('S', inplace=True)
test["Embarked"].fillna('S', inplace=True)

embarked_train_dummies = pd.get_dummies(train['Embarked'])
embarked_test_dummies = pd.get_dummies(test['Embarked'])

embarked_train_dummies.columns = ['S', 'C', 'Q']
embarked_test_dummies.columns = ['S', 'C', 'Q']

train.drop(['Embarked'], axis=1, inplace=True)
test.drop(['Embarked'], axis=1, inplace=True)

train = train.join(embarked_train_dummies)
test = test.join(embarked_test_dummies)

X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,
        train_size=0.8,shuffle=True, random_state=31)

n_splits=5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=31)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

parameters=[
    {'n_estimators':[100,200],'max_depth':[6,8,10,23]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10],
     'n_jobs':[-1,2,4]},
]

#2. 모델구성
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV

model=HalvingGridSearchCV(RandomForestClassifier(),parameters,cv=kfold, verbose=1,
                   refit=True, n_jobs=-1)

#3. 컴파일, 훈련
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
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
y_pred_best=model.best_estimator_.predict(x_test)
print("최적 튠 acc : ",accuracy_score(y_test,y_pred_best))
print('걸린시간:',np.round(end-start,2))

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestClassifier(max_depth=8, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 8, 'n_estimators': 200}
# best_score:  0.8426671919629666
# model.score: 0.8156424581005587
# acc score : 0.8156424581005587
# 최적 튠 acc :  0.8156424581005587
# 걸린시간: 7.37

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=5, min_samples_split=10, n_jobs=2)
# 최적의 파라미터:  {'n_jobs': 2, 'min_samples_split': 10, 'min_samples_leaf': 5}
# best_score:  0.8314882300797792
# model.score: 0.8044692737430168
# acc score : 0.8044692737430168
# 최적 튠 acc :  0.8044692737430168

# HalvingGridSearchCV
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 26
# max_resources_: 712
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 56
# n_resources: 26
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# ----------
# iter: 1
# n_candidates: 19
# n_resources: 78
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 234
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 702
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=5, n_jobs=4)최적의 파라미터:  {'min_samples_leaf': 5, 'min_samples_split': 2, 'n_jobs': 4}
# best_score:  0.8314285714285713
# model.score: 0.7821229050279329
# acc score : 0.7821229050279329
# 최적 튠 acc :  0.7821229050279329
# 걸린시간: 9.96