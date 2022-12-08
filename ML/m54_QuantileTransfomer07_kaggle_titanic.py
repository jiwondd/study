import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
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

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=123)

mm=MinMaxScaler() 
stan=StandardScaler()
ma=MaxAbsScaler()
robus=RobustScaler() 
quan=QuantileTransformer()
power_yeo=PowerTransformer(method='yeo-johnson') 
power_box=PowerTransformer(method='box-cox')

scalers=[mm,stan,ma,robus,quan,power_yeo,power_box]
for scaler in scalers:
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        result=accuracy_score(y_test,y_predict)
        scale_name=scaler.__class__.__name__
        print('{0}결과:{1:4f}'.format(scale_name,result))


# MinMaxScaler결과:0.832402
# StandardScaler결과:0.821229
# MaxAbsScaler결과:0.832402
# RobustScaler결과:0.826816
# The Box-Cox transformation can only be applied to strictly positive data
