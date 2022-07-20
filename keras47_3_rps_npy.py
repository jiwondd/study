from pickletools import optimize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# cats = 1 / dog = 2

# 1. 데이터

x_data=np.load('d:/study_data/_save/_npy/keras47_3_train_x.npy')
y_data=np.load('d:/study_data/_save/_npy/keras47_3_train_y.npy')
# x_test=np.load('d:/study_data/_save/_npy/keras47_1_test_x.npy')
# y_test=np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=42)


print(x_train.shape)    #(14, 200, 200, 3)
print(y_train.shape)    #(14, 3)
print(x_test.shape)     #(6, 200, 200, 3)
print(y_test.shape)     #(6, 3)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(32,(2,2), input_shape=(200,200,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(3,activation='softmax'))
# model.summary()


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
star_time=time.time()
hist=model.fit(x_train,y_train,epochs=100,steps_per_epoch=200,
          validation_split=0.2,validation_steps=4)
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

# loss: 0.7461754679679871
# val_loss: 1.0175644159317017
# accuracy: 0.8399999737739563
# val_accuracy: 0.3333333432674408
# 걸린시간: 23.152841329574585