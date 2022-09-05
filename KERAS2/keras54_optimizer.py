import numpy as np

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,3,5,4,7,6,7,11,9,7])

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model=Sequential()
model.add(Dense(1000,input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# 3. 컴파일, 훈련
# from keras.optimizers import Adam, Adadelta, Adagrad,Adamax
# from keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam
# from tensorflow.python.keras.optimizer_v1 import RMSprop, SGD, Nadam

# learning_rate=0.1
# optimizer=adam.Adam(learning_rate=learning_rate)
# optimizer=adadelta.Adadelta(learning_rate=learning_rate)
# optimizer=adagrad.Adagrad(learning_rate=learning_rate)
# optimizer=adamax.Adamax(learning_rate=learning_rate)
# optimizer=rmsprop.RMSprop(learning_rate=learning_rate)
# optimizer=nadam.Nadam(learning_rate=learning_rate)

# model.compile(loss='mse',optimizer=optimizer)
# model.fit(x,y,epochs=50,batch_size=1)

# # 4. 평가, 예측
# loss=model.evaluate(x,y)
# y_pred=model.predict([11])

# print('loss:',round(loss,4),'lr:',learning_rate,'결과물:',y_pred)

learning_rate = 0.001

optlist = [adam.Adam, adadelta.Adadelta, adagrad.Adagrad, adamax.Adamax, rmsprop.RMSprop, nadam.Nadam]
for optimizer in optlist:
    optimizer = optimizer(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x, y, epochs=50, batch_size=1, verbose=0)

    # 4. evaluate, predict
    loss = model.evaluate(x, y)
    y_pred = model.predict([11])

    print('optimizer: ', optimizer.__class__.__name__, '   /loss: ', round(loss, 4), '/lr: ', learning_rate, '/predict result: ', y_pred)
    
    
# optimizer:  Adam    /loss:  3.2216 /lr:  0.0001 /predict result:  [[12.640004]]
# optimizer:  Adadelta    /loss:  2.947 /lr:  0.0001 /predict result:  [[12.36316]]
# optimizer:  Adagrad    /loss:  2.3084 /lr:  0.0001 /predict result:  [[10.972901]]
# optimizer:  Adamax    /loss:  2.2376 /lr:  0.0001 /predict result:  [[10.727072]]
# optimizer:  RMSprop    /loss:  2.2221 /lr:  0.0001 /predict result:  [[10.483887]]
# optimizer:  Nadam    /loss:  2.253 /lr:  0.0001 /predict result:  [[11.087538]]

# optimizer:  Adam    /loss:  2.1654 /lr:  0.001 /predict result:  [[10.640084]]
# optimizer:  Adadelta    /loss:  2.1623 /lr:  0.001 /predict result:  [[10.593489]]
# optimizer:  Adagrad    /loss:  2.1606 /lr:  0.001 /predict result:  [[10.43174]]
# optimizer:  Adamax    /loss:  2.1612 /lr:  0.001 /predict result:  [[10.398475]]
# optimizer:  RMSprop    /loss:  21.54 /lr:  0.001 /predict result:  [[18.226768]]
# optimizer:  Nadam    /loss:  4.2305 /lr:  0.001 /predict result:  [[7.9063025]]
