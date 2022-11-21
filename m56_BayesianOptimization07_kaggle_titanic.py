import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from icecream import ic

# 1. 데이터
path='./_data/kaggle_titanic/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv')
train = train_set.drop(['PassengerId','Name', 'Ticket','Cabin','SibSp'], axis = 1 )
test = test_set.drop(['Name', 'Ticket','Cabin','SibSp'], axis= 1)

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

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=123,train_size=0.8)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

bayesian_params={
                 'max_depth':[1,6],
                 'min_child_weight':[0.1,5],
                 'reg_alpha':[0,10],
                 'reg_lambda':[0,10]
                }

def lgb_hamsu(max_depth,min_child_weight,
              reg_lambda,reg_alpha):
    params={
             'n_estimators':200,"learning_rate":0.02,
             'max_depth':int(round(max_depth)),
             'min_child_weight':int(round(min_child_weight)),
             'reg_lambda':max(reg_lambda,0),
             'reg_alpha':max(reg_alpha,0)
    }
    
    model=XGBClassifier(**params) # **키워드 받을게(딕셔너리형태) *여러개의 인자를 받을게(1개도되고 여러개도되고)
    model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='error',
              verbose=0,
              early_stopping_rounds=50)
    y_pred=model.predict(x_test)
    result=accuracy_score(y_test,y_pred)
    
    return result

lgb_bo=BayesianOptimization(f=lgb_hamsu,
                            pbounds=bayesian_params,
                            random_state=123)

lgb_bo.maximize(init_points=2,n_iter=20)

print(lgb_bo.max)