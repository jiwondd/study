from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
print(x)
print(y)
print(x.shape,y.shape) #(20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=72)

#2. 모델구성
model=Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train, epochs=300, batch_size=50)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)

# loss: 0.5962857604026794
# r2스코어: 0.5464863419455076
# rs 0.55~0.58
