from pickletools import optimize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib
import matplotlib.pyplot as plt
# 1 -> 정상 / 0 -> 이상있음

# 1. 데이터

x_train=np.load('d:/study_data/_save/_npy/keras46_5_train_x.npy')
y_train=np.load('d:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test=np.load('d:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test=np.load('d:/study_data/_save/_npy/keras46_5_test_y.npy')

print(x_train) 
print(x_train.shape)    #(160, 150, 150, 1)
print(y_train.shape)    #(160,)
print(x_test.shape)     #(120, 150, 150, 1)
print(y_test.shape)     #(120,)

'''
# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(32,(2,2), input_shape=(100,100,1),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1]) 배치를 최대로 잡으면 이렇게도 가능하다
star_time=time.time()
hist=model.fit_generator(xy_train, epochs=10, steps_per_epoch=32, #<-전체 데이터/batch = 160/5 = 32
                    validation_data=xy_test,
                    validation_steps=4)
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

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') 
plt.plot(hist.history['val_loss'],marker='.',c='green',label='val_loss')
plt.plot(hist.history['accuracy'],marker='.',c='yellow',label='accuracy')
plt.plot(hist.history['val_accuracy'],marker='.',c='purple',label='val_acc')
plt.grid() #모눈종이ㄱ
plt.title('loss and val_loss') #제목
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()

# loss: 0.6933607459068298
# val_loss: 0.6931483149528503
# accuracy: 0.4625000059604645
# val_accuracy: 0.5
# 걸린시간: 32.34719657897949 <-gpu

# loss: 0.29662764072418213
# val_loss: 425.73876953125
# accuracy: 0.8812500238418579
# val_accuracy: 0.30000001192092896
# 걸린시간: 242.5329475402832 <-cpu
'''
