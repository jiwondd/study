import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from csv import reader
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler,StandardScaler

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

x_train = train.drop("Survived",axis=1)
y = train["Survived"]
x_test  = test.drop("PassengerId",axis=1).copy()

# print(np.unique(y, return_counts=True)) 
# (array([0, 1], dtype=int64), array([549, 342], dtype=int64))

lda=LinearDiscriminantAnalysis() 
lda.fit(x_train,y)
x=lda.transform(x_train)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
model=XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
results=model.score(x_test,y_test)
print('결과:',results)
print('걸린시간:',np.round(end-start,2))

# 결과: 0.8603351955307262
# 걸린시간: 0.42