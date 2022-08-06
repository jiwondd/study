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

# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 분류임 어그로...(논리적인회귀=분류)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier

model=LinearSVC()
model1=LogisticRegression()
model2=KNeighborsClassifier()
model3=DecisionTreeClassifier()
model4=RandomForestClassifier()

#3. 컴파일, 훈련
model.fit(x_train,y_train)
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)


#4. 평가, 예측
result=model.score(x_test,y_test)
print('LinearSVC 결과: ',result)
y_predict1=model.predict(x_test)
acc=accuracy_score(y_test,y_predict1)
print('LinearSVC score :', acc) # =결과 result 
print('=========================================')

result=model1.score(x_test,y_test)
print('LogisticRegression 결과: ',result)
y_predict1=model1.predict(x_test)
acc=accuracy_score(y_test,y_predict1)
print('LogisticRegression_acc score :', acc) # =결과 result 
print('=========================================')

result=model2.score(x_test,y_test)
print('KNeighborsClassifier 결과: ',result)
y_predict2=model2.predict(x_test)
acc=accuracy_score(y_test,y_predict2)
print('KNeighborsClassifier_acc score :', acc) # =결과 result 
print('=========================================')

result=model3.score(x_test,y_test)
print('DecisionTreeClassifier 결과: ',result)
y_predict3=model3.predict(x_test)
acc=accuracy_score(y_test,y_predict3)
print('DecisionTreeClassifier_acc score :', acc) # =결과 result 
print('=========================================')

result=model4.score(x_test,y_test)
print('RandomForestClassifier 결과: ',result)
y_predict4=model4.predict(x_test)
acc=accuracy_score(y_test,y_predict4)
print('RandomForestClassifier_acc score :', acc) # =결과 result 

# LinearSVC 결과:  0.7150837988826816
# LinearSVC score : 0.7150837988826816
# =========================================
# LogisticRegression 결과:  0.7821229050279329
# LogisticRegression_acc score : 0.7821229050279329
# =========================================
# KNeighborsClassifier 결과:  0.6759776536312849
# KNeighborsClassifier_acc score : 0.6759776536312849
# =========================================
# DecisionTreeClassifier 결과:  0.7653631284916201
# DecisionTreeClassifier_acc score : 0.7653631284916201
# =========================================
# RandomForestClassifier 결과:  0.776536312849162
# RandomForestClassifier_acc score : 0.776536312849162
