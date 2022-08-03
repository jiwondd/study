from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=72)

#2. 모델구성

model=LinearSVC()

#3.컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
print('결과: ',result)
y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc) # =결과 result 


# loss: 1080.822998046875
# r2스코어: 0.746747006658822

# 결과:  0.015037593984962405
# acc score : 0.015037593984962405 <-LinearSVC