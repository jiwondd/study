import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])

# 2. 모델
model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()
print(model.weights)
print('===================================================================')
print(model.trainable_weights)
print('===================================================================')
print(len(model.weights)) #6
print(len(model.trainable_weights)) #6
# model.trainable=False
print('===================================================================')
print(len(model.weights)) #6
print(len(model.trainable_weights)) #0
print('===================================================================')
# print(model.trainable_weights) #[]
# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0
# _________________________________________________________________
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.5722169 , -1.122747  ,  0.49328876]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <바이어스1>
# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy= array([[ 0.82076013, -0.85751337],[-0.44686955, -0.2677331 ],[ 0.39269936,  0.3420012 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32,
# numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, 
# numpy=array([[0.05845499],[1.3128098 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=1,epochs=100)
y_predict=model.predict(x)
# print(y_predict[:3]) #[[0.6361715] [1.272343 ] [1.9085147]] model.trainable=False
# model.trainable=True [[1.268609 ] [2.1635807] [3.058552 ]]
# 전이학습할때 인풋, 아웃풋 부분만 내 데이터로 커스터마이징하고 중간레이어는 trainable=False로 잡으면 가중치는 고정