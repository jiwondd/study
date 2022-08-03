from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

#2. 모델구성
model=LinearSVR()

#3.컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('결과: ',result)

'''

loss: 0.43223631381988525
r2스코어: 0.6701515707751251 <-랜덤스테이트 42

r2스코어: 0.059038155927817804
결과:  0.059038155927817804 <-LinearSVR



'''