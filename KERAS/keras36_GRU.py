import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU

# 1. 데이터
datasets=np,np.array([1,2,3,4,5,6,7,8,9,10])
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y=np.array([4,5,6,7,8,9,10])

# RNN = input_shape=(행,열,몇개씩 자르는지!)
print(x.shape,y.shape) #(7, 3) (7,)

x=x.reshape (7,3,1)
print(x.shape) #(7,3,1)



# 2. 모델구성
model=Sequential()                                                   
model.add(GRU(units=10,input_shape=(3,1)))
# GRU = units : 10 -> 3*10*(1+1+10)=360 / SimpleRNN * 3 = GRU
model.add(Dense(5))
model.add(Dense(1))
model.summary()





'''
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(326,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000)

# 4. 평가, 예측
loss=model.evaluate(x,y)
y_pred=np.array([8,9,10]).reshape(1,3,1)  #[[[8],[9],[10]]]
result=model.predict(y_pred)
print('loss:',loss)
print('[8,9,10]의 결과:',result)

# loss: 2.599016852400382e-06
# [8,9,10]의 결과: [[11.007277]] <-LSTM

# loss: 2.2582753445021808e-08
# [8,9,10]의 결과: [[10.877161]] <-GRU
'''
