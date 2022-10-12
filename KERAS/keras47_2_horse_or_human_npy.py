from pickletools import optimize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# cats = 1 / dog = 2

# 1. 데이터

x_data=np.load('d:/study_data/_save/_npy/keras47_2_train_x.npy')
y_data=np.load('d:/study_data/_save/_npy/keras47_2_train_y.npy')
# x_test=np.load('d:/study_data/_save/_npy/keras47_1_test_x.npy')
# y_test=np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=42)


print(x_train.shape)    #(70, 200, 200, 3)
print(y_train.shape)    #(70,)
print(x_test.shape)     #(70, 200, 200, 3)
print(y_test.shape)     #(30,)


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

# loss: 0.6753200888633728        
# val_loss: 0.694843053817749     
# accuracy: 0.6000000238418579    
# val_accuracy: 0.5306122303009033
# 걸린시간: 42.99685049057007 
