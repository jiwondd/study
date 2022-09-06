import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D
#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델구성
activation='relu'
optimizer='adam'
drop=0.2

inputs = Input(shape=(28,28,1), name='input')
x = Conv2D(64, (2,2),padding='valid',
           activation=activation, name='hidden1')(inputs)
x = Dropout(drop)(x)
# x = Conv2D(64, (2,2),padding='same',
#            activation=activation, name='hidden2')(inputs)
# x = Dropout(drop)(x)
# x = MaxPooling2D()(x)
# x = Conv2D(32, (3,3),padding='valid',
#            activation=activation, name='hidden3')(inputs)
# x = Dropout(drop)(x)
# x = Flatten()(x) #(25*25*32)
x = GlobalAveragePooling2D()(x)

x = Dense(100, activation=activation, name='hidden4')(x) #20000*256 <-연상양이 너무 많아져서 터진다
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
#_________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input (InputLayer)          [(None, 28, 28, 1)]       0

#  hidden3 (Conv2D)            (None, 26, 26, 32)        320

#  dropout_2 (Dropout)         (None, 26, 26, 32)        0

#  flatten (Flatten)           (None, 21632)             0

#  hidden4 (Dense)             (None, 100)               2163300

#  dropout_3 (Dropout)         (None, 100)               0

#  outputs (Dense)             (None, 10)                1010

# =================================================================
# Total params: 2,164,630
# Trainable params: 2,164,630
# Non-trainable params: 0
# _________________________________________________________________
# 연산양의 차이를 보자
#  global_average_pooling2d (G  (None, 32)               0
#  lobalAveragePooling2D)

#  hidden4 (Dense)             (None, 100)               3300

#  dropout_3 (Dropout)         (None, 100)               0

#  outputs (Dense)             (None, 10)                1010

# =================================================================
# Total params: 4,630
# Trainable params: 4,630
# Non-trainable params: 0
# _________________________________________________________________

model.compile(optimizer=optimizer, metrics=['acc'],
                loss='sparse_categorical_crossentropy')
    
    
import time
start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.4,batch_size=128)
end = time.time()

from sklearn.metrics import accuracy_score
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('걸린시간 : ', end - start)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))

# 걸린시간 :  23.02012038230896
# loss :  1.3952
# accuracy :  0.4813