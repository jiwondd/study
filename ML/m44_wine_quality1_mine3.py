import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from cProfile import label
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

#.1 데이터
path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=0, sep=';')
# print(data_set.shape) (4898, 11)                             ㄴ기준으로 컬럼을 나눠줘

#.1 데이터
y=data_set['quality']
x=data_set.drop(['quality'],axis=1)

le=LabelEncoder()
y=le.fit_transform(y)

scaler=MinMaxScaler()
scaler.fit(x)
data_set=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,
                                                  random_state=123,shuffle=True)

kFold=StratifiedKFold(n_splits=2, shuffle=True,random_state=123)

from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 이진분류임 어그로...(논리적인회귀=분류)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

model=LinearSVC()
model1=LogisticRegression()
model2=KNeighborsClassifier()
model3=DecisionTreeClassifier()
model4=RandomForestClassifier()

#3.컴파일, 훈련
model.fit(x_train,y_train)
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict)
print('LinearSVC acc스코어:',acc)
print('LinearSVC 결과: ',result)
print('=========================================')
result=model1.score(x_test,y_test)
y_predict1=model1.predict(x_test)
acc=accuracy_score(y_test,y_predict1)
print('LogisticRegression acc:',acc)
print('LogisticRegression 결과: ',result)
print('=========================================')
result=model2.score(x_test,y_test)
y_predict2=model2.predict(x_test)
acc=accuracy_score(y_test,y_predict2)
print('KNeighborsClassifier acc:',acc)
print('KNeighborsClassifier 결과: ',result)
print('=========================================')
result=model3.score(x_test,y_test)
y_predict3=model3.predict(x_test)
acc=accuracy_score(y_test,y_predict3)
print('DecisionTreeClassifier acc:',acc)
print('DecisionTreeClassifier 결과: ',result)
print('=========================================')
result=model4.score(x_test,y_test)
y_predict4=model4.predict(x_test)
acc=accuracy_score(y_test,y_predict4)
print('RandomForestClassifier acc:',acc)
print('RandomForestClassifier 결과: ',result)

# LinearSVC acc스코어: 0.37244897959183676
# LinearSVC 결과:  0.37244897959183676
# =========================================
# LogisticRegression acc: 0.4846938775510204
# LogisticRegression 결과:  0.4846938775510204
# =========================================
# KNeighborsClassifier acc: 0.49795918367346936
# KNeighborsClassifier 결과:  0.49795918367346936
# =========================================
# DecisionTreeClassifier acc: 0.6061224489795919
# DecisionTreeClassifier 결과:  0.6061224489795919
# =========================================
# RandomForestClassifier acc: 0.6959183673469388
# RandomForestClassifier 결과:  0.6959183673469388
