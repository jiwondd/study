from pickletools import optimize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib
import matplotlib.pyplot as plt
# 1 -> 정상 / 0 -> 이상있음

# 1. 데이터
train_datagen=ImageDataGenerator(
    rescale=1./255, #MinMaxs
    horizontal_flip=True,   #수평반전
    vertical_flip=True,     #수직반전
    width_shift_range=0.1,  #좌우이동
    height_shift_range=0.1, #상하이동
    rotation_range=5,       #임의의 각도록 회전시킨다.
    zoom_range=1.2,         #확대범위
    shear_range=0.7,        #기울기(?)/찌그러트리면서 돌리는?
    fill_mode='nearest'     #여기까지 다 넣어도 랜덤으로 선택되서 조정됨
)

test_datagen=ImageDataGenerator(
    rescale=1./255           #평가데이터는 손대지 않고 그 데이터 그대로 해야한다.
)
#                       폴더에서 불러오기ㄱ
xy_train=train_datagen.flow_from_directory(
    'd:/_data/image/brain/train/',
    target_size=(100,100),  #지정사이즈 -> 이미지 수집할때 각자 다 다른 사이즈의 이미지를 임의로 사이즈를 맞춘다
    batch_size=5,
    class_mode='categorical',    
    color_mode='grayscale', 
    shuffle=True,
    )                    # Found 160 images belonging to 2 classes.

xy_test=test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(100,100),
    batch_size=5,
    class_mode='categorical', #test에 ad/normal 두가지 분류니까 만약 3가지 이상이면 categorical
    color_mode='grayscale',
    shuffle=True,
    )          #Found 120 images belonging to 2 classes.

print(xy_train[0][0])  
print(xy_train[0][1])
print(xy_train[31][0].shape,xy_train[31][1].shape) #(5, 100, 100, 1) (5, 2)

print(type(xy_train))
print(type(xy_train[0]))        #<class 'tuple'>
print(type(xy_train[0][0]))     #<class 'numpy.ndarray'>     /  numpy 두개가 튜플로 묶여있다. 

# 현재 5,200,200,1 자리 데이터가 32덩어리 있다.

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(32,(2,2), input_shape=(100,100,1),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(2,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1]) 배치를 최대로 잡으면 이렇게도 가능하다
star_time=time.time()
hist=model.fit_generator(xy_train, epochs=100, steps_per_epoch=32, #<-전체 데이터/batch = 160/5 = 32
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

# loss: 0.2855755090713501
# val_loss: 0.2608962655067444
# accuracy: 0.887499988079071
# val_accuracy: 0.8999999761581421
# 걸린시간: 31.294046878814697  <-categorical 로 변경 
