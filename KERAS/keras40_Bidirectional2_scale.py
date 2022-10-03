import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.layers import Bidirectional

# 1. 데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict=np.array([50,60,70])  # 80을 예측해보자!

print(x.shape,y.shape) #(13, 3) (13,)
x=x.reshape (13,3,1)
print(x.shape,y.shape) #(13, 3, 1) 
print(x_predict.shape)



# 2. 모델구성
model=Sequential()                                                   
model.add(Bidirectional(SimpleRNN(10,return_sequences=True,activation='relu'),input_shape=(3,1)))
model.add(SimpleRNN(10,return_sequences=True))
model.add(Bidirectional(SimpleRNN(10))) #<-여기다가 리턴 시퀀스 넣으니까 3차원이 밑으로 쭉쭉 떨어져서 값이 3개 나옴^^
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
# model.summary()
model.add(Dense(326,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

# 덴스로 바뀌는 시점에는 리턴시퀀스 쓰면 안된다

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

# 4. 평가, 예측
loss=model.evaluate(x,y)
y_pred=x_predict.reshape(1,3,1)
print(x_predict)
result=model.predict(y_pred)
print('loss:',loss)
print('[50,60,70]의 결과:',result)

# loss: 0.002671613125130534
# [50,60,70]의 결과: [[78.310165]]

# loss: 0.00011626197374425828
# [50,60,70]의 결과: [[77.240555]]

