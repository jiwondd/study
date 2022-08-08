import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score

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

x_train = train.drop("Survived",axis=1)
y_train = train["Survived"]
x_test  = test.drop("PassengerId",axis=1).copy()

# x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
#         train_size=0.8,shuffle=True, random_state=31)
n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 분류임 어그로...(논리적인회귀=분류)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier

model=LinearSVC()

# 3.4. 컴파일 훈련 평가 예측
scores=cross_val_score(model,x_train,y_train,cv=kfold)
# scores=cross_val_score(model,x,y,cv=5) 위에랑 똑같아요~
print('ACC:',scores,'\n cross_val_score:', round(np.mean(scores),4))

# ACC: [0.73743017 0.80898876 0.69101124 0.74719101 0.50561798]
#  cross_val_score: 0.698
