
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=MinMaxScaler()
scaler=StandardScaler()
scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
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


# loss: 0.19975309073925018
# accuracy: 0.920501172542572
# ===================================
# acc score : 1.0<-RobustScaler

# LinearSVC 결과:  0.711220880700154
# LinearSVC score : 0.711220880700154
# =========================================
# LogisticRegression 결과:  0.7222963262566371
# LogisticRegression_acc score : 0.7222963262566371
# =========================================
# KNeighborsClassifier 결과:  0.9273856957221415
# KNeighborsClassifier_acc score : 0.9273856957221415
# =========================================
# DecisionTreeClassifier 결과:  0.9385730144660637
# DecisionTreeClassifier_acc score : 0.9385730144660637
# =========================================
# RandomForestClassifier 결과:  0.9555433164376135
# RandomForestClassifier_acc score : 0.9555433164376135


