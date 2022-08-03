from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

#2. 모델구성
model=LinearSVR()


#3.컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('결과: ',result)
# loss: 17.545747756958008
# r2스코어: 0.7876257019581974

# r2스코어: 0.6244182202889601
# 결과:  0.6244182202889601