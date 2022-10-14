import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.covariance import EllipticEnvelope
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# 1. 데이터
path='./_data/kaggle_titanic/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv')

print(train_set.info())

# outliers=EllipticEnvelope(contamination=.1)
# outliers.fit(train_set)
# results=outliers.predict(train_set)
# print(results)

def getZscoreOutlier(df,col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:",out)
    print("min",np.median(out))
    return np.median(out)

col = "Age"
minOutlier = getZscoreOutlier(train_set,col)
print(train_set[train_set[col] >= minOutlier])

col = "Fare"
minOutlier = getZscoreOutlier(train_set,col)
print(train_set[train_set[col] >= minOutlier])

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(x = train_set['Age'], y = train_set['Survived'])
# plt.xlabel('Age', fontsize = 13)
# plt.ylabel('Survived', fontsize = 13)
# plt.show()

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
test  = test.drop("PassengerId",axis=1).copy()

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,
        train_size=0.8,shuffle=True, random_state=123)

print(x_train.shape)

#2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model=RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)


# 4.평가, 예측
results=model.score(x_test,y_test)
print('결과:',results)


# 결과: 0.8379888268156425 중위값
# 결과: 0.8379888268156425 평균값

# 랜덤포레스트
# 결과: 0.8324022346368715 평균값
# 결과: 0.8435754189944135 중위값
# 결과: 0.8324022346368715 최소값
# 결과: 0.8324022346368715 최대값