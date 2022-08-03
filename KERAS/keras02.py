import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3,5,4])
y=np.array([1,2,3,4,5])

#2.모델구성
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense             

model=Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=500)

#4. 평가, 예측
loss=model.evaluate(x,y)
print('loss:',loss)

result=model.predict([6])
print('6의 예측값:',result)

# loss: 0.40279874205589294
# 6의 예측값: [[5.972604]]

