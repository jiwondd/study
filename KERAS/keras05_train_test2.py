import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])
# x_train=np.array([1,2,3,4,5,6,7])
# y_train=np.array([1,2,3,4,5,6,7])
#x_test=np.array([8,9,10])
#y_test=np.array([8,9,10])

#[과제] 넘파이 리스트의 슬라이싱!! 7:3으로 자르기
x_train=x[0:7]
x_test=x[8:10]
y_train=y[0:7]
y_test=y[8:10]

print(x_train) #[1 2 3 4 5 6 7]
print(y_train) #[1 2 3 4 5 6 7]
print(x_test) #[ 9 10]
print(y_test) #[ 9 10]

'''
#2. 모델
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_train,y_train)
print('loss:',loss)
result=model.predict([11])
print('11의 예측값:',result)

'''

