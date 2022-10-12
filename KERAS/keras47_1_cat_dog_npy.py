# 넘파이 불러와서 모델링!!
from pickletools import optimize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib
import matplotlib.pyplot as plt

# cats = 1 / dog = 2

# 1. 데이터

x_train=np.load('d:/study_data/_save/_npy/keras47_1_train_x.npy')
y_train=np.load('d:/study_data/_save/_npy/keras47_1_train_y.npy')
x_test=np.load('d:/study_data/_save/_npy/keras47_1_test_x.npy')
y_test=np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')

print(x_train.shape)    #(100, 200, 200, 3)
print(y_train.shape)    #(100,)
print(x_test.shape)     #(100, 200, 200, 3)
print(y_test.shape)     #(100,)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(32,(2,2), input_shape=(200,200,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))
# model.summary()


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
star_time=time.time()
hist=model.fit(x_train,y_train,epochs=100,steps_per_epoch=80,
          validation_split=0.7,validation_steps=4)
accuracy=hist.history['accuracy']


val_accuracy=hist.history['val_accuracy']
loss=hist.history['loss']
val_loss=hist.history['val_loss']
end_time=time.time()-star_time

print('loss:',loss[-1])
print('val_loss:',val_loss[-1])
print('accuracy:',accuracy[-1])
print('val_accuracy:',val_accuracy[-1])
print('걸린시간:',end_time)

# loss: 1.5133151691770763e-06
# val_loss: 7.137775897979736
# accuracy: 1.0
# val_accuracy: 0.6285714507102966
# 걸린시간: 55.751386642456055
