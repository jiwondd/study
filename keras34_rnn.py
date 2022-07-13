import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN

# 1. 데이터
datasets=np,np.array([1,2,3,4,5,6,7,8,9,10])
# y = ?? 
# 나중에는 함수로 알려줄거니까 그때는 함수 이용해서 자르면 됩니당
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y=np.array([4,5,6,7,8,9,10])

# RNN = input_shape=(행,열,몇개씩 자르는지!)
print(x.shape,y.shape) #(7, 3) (7,)

x=x.reshape (7,3,1)
print(x.shape) #(7,3,1)



# 2. 모델구성
model=Sequential()
model.add(SimpleRNN(10,input_shape=(3,1)))
# model.add(SimpleRNN(8))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
model.summary()


'''
# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000)

# 4. 평가, 예측
loss=model.evaluate(x,y)
y_pred=np.array([8,9,10]).reshape(1,3,1)  #[[[8],[9],[10]]]
result=model.predict(y_pred)
print('loss:',loss)
print('[8,9,10]의 결과:',result)

# loss: 1.2018322535084502e-12
# [8,9,10]의 결과: [[10.826964]]
'''