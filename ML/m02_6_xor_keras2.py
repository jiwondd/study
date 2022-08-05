# 단층 퍼셉트론
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 1.데이터
x_data=[[0,0],[0,1],[1,0],[1,1]] # ( 4 , 2 )
y_data=[0,1,1,0] #(4, )

# 2.모델구성
# model=LinearSVC()
# model=Perceptron()
model=Sequential()
model.add(Dense(16,input_dim=2))
model.add(Dense(32,activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3.훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data,y_data,batch_size=1,epochs=100)

# 4. 평가, 예측
y_predict=model.predict(x_data)
print(x_data,'의 예축결과:',y_predict)
result=model.evaluate(x_data,y_data)
print('metrics score:',result[1])

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예축결과: [[1.9949405e-03]
#  [9.9834108e-01]
#  [9.9879611e-01]
#  [8.0286706e-04]]
# 1/1 [==============================] - 0s 118ms/step - loss: 0.0014 - acc: 
# 1.0000
# metrics score: 1.0