from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x=np.array(range(1,17))
y=np.array(range(1,17))


#[실습] train_test_split로만 나누기 10 / 3 / 3
x_train,x_test,y_train,y_test=train_test_split(x,y,
        test_size=0.2, random_state=31)
# x_test,x_val,y_test,y_val=train_test_split(x_test_val,y_test_val,
#         train_size=0.5, random_state=72)

# print(x_train) #[ 5 16 12  8 14 13  6 11  4  9]
# print(y_train) #[ 5 16 12  8 14 13  6 11  4  9]
# print(x_test) #[15  7  3]
# print(y_test) #[15  7  3]
# print(x_val) #[ 1  2 10]
# print(y_val) #[ 1  2 10]


#2. 모델
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.25)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
result=model.predict([17])
print("17의 예측값:",result)

# loss: 0.023004189133644104
# 17의 예측값: [[16.849424]]
