from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape,y.shape) #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)

#[실습]
#R2 0.62 이상

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.9,shuffle=True, random_state=31)

#2. 모델구성
model=Sequential()
model.add(Dense(5,input_dim=10))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train, epochs=300, batch_size=10,validation_split=0.1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)

# loss: 1080.822998046875
# r2스코어: 0.746747006658822
'''
loss: 2633.421875
r2스코어: 0.5133393119860061 train 0.9 / val 0.3

loss: 2668.4443359375
r2스코어: 0.5068670136409048 train 0.9 / val 0.2

loss: 2681.824951171875
r2스코어: 0.5043942801890218 train 0.9 / val 0.1

loss: 2875.76611328125
r2스코어: 0.5686935844813663 train 0.7 / val 0.1 / random 31

loss: 2335.279052734375
r2스코어: 0.5974426324664464 train 0.9 / val 0.1






'''
