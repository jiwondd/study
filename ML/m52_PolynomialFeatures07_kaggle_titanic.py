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

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score

model=make_pipeline(StandardScaler(),
                    DecisionTreeClassifier()
                    )


model.fit(x_train,y_train)

score=model.score(x_test,y_test)
cv_score=cross_val_score(model,x_train, y_train, cv=kfold)
ic(score) 
ic(cv_score,np.mean(cv_score))

############polynomial적용후###############
pf=PolynomialFeatures(degree=2) #include_bias=False
xp=pf.fit_transform(X_train)

x_train2,x_test2,y_train2,y_test2=train_test_split(xp,Y_train,
        train_size=0.8,shuffle=True, random_state=1234)

# 2.모델
model2=make_pipeline(StandardScaler(),
                    DecisionTreeClassifier()
                    )

model.fit(x_train2,y_train2)

poly_score=model.score(x_test2,y_test2)
cv_score2=cross_val_score(model2,x_train2, y_train2, cv=kfold)
ic(poly_score)
ic(cv_score2,np.mean(cv_score2))

# ic| score: 0.8268156424581006
# ic| cv_score: array([0.84615385, 0.74125874, 0.77464789, 0.76056338, 0.78169014])
#     np.mean(cv_score): 0.7808627991726583

# ic| poly_score: 0.7932960893854749
# ic| cv_score2: array([0.79020979, 0.74825175, 0.73239437, 0.80985915, 0.76760563])
#     np.mean(cv_score2): 0.7696641386782231
