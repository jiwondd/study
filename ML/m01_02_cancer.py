import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target


#2. 모델구성
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

model=LinearSVC()


#3.컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
print('결과: ',result)
y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc) # =결과 result

'''
loss :  [0.17140837013721466, 0.9473684430122375]
acc score : 0.9473684210526315 

LinearSVC
결과:  0.9473684210526315
acc score : 0.9473684210526315
'''



