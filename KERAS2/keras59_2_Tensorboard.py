import numpy as np
from keras.datasets import mnist
import tensorboard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D
#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 padding='same',
                 input_shape=(28,28,1)))
model.add(Conv2D(32,(2,2),
                 padding='valid', 
                 activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(16))
model.add(Dense(8,activation='relu'))
model.add(Dense(10,activation='softmax')) 

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard
from tensorflow.keras.optimizers import Adam
import time

es=EarlyStopping(monitor='val_loss',patience=15,mode='min',verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auteo',verbose=1,
                            factor=0.5)
tb=TensorBoard(log_dir='d:/study_data/tensorboard_log/_graph',histogram_freq=0,
               write_graph=True,write_images=True)
# 실행방법 : tensorboard --logdir=.
# http://localhost:6006
# http://127.0.0.1:6006

learnig_rate=0.01
optimizer=Adam(lr=learnig_rate)
model.compile(optimizer=optimizer, metrics=['acc'],
                loss='sparse_categorical_crossentropy')
start=time.time()
hist=model.fit(x_train,y_train, epochs=100,batch_size=32,verbose=1,
               callbacks=[es,reduce_lr,tb], validation_split=0.2)
end=time.time()

loss, acc = model.evaluate(x_test, y_test)

print('learning_raed:',learnig_rate)
print('걸린시간 : ', round(end - start,4))
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))

# learning_raed: 0.01
# 걸린시간 :  213.3723
# loss :  2.301
# accuracy :  0.1135

import matplotlib.pyplot as plt
# 1
plt.figure(figsize=(9,5))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'],marker='.',c='red',label='acc')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc','val_loss'])

plt.show()