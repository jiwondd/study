from calendar import c
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test)= fashion_mnist.load_data()

train_datagen=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
train_datagen2=ImageDataGenerator(
    rescale=1./255
)

augument_size=40000             # 0~60000(59999) 사이의 값 ㄱ
batch_size=64

randidx=np.random.randint(x_train.shape[0],size=augument_size) # ( 60000 , 40000 )
print(x_train.shape) #(60000, 28, 28)
print(randidx) #[41376  9918 23512 ...  7958 10832 39610]
print(np.min(randidx),np.max(randidx)) # 2 59999 -> 3 59998  /
print(type(randidx)) #<class 'numpy.ndarray'>

x_argumented=x_train[randidx].copy()
y_argumented=y_train[randidx].copy()
#                               ㄴ 같은 값을 덮어 버리지 않도록 옆쪽에 복사본을 만드는것
print(x_argumented.shape) #(40000, 28, 28)
print(y_argumented.shape) #(40000,)  

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_argumented=x_argumented.reshape(x_argumented.shape[0],x_argumented.shape[1],x_argumented.shape[2],1)

x_argumented=train_datagen.flow(x_argumented,y_argumented, 
                                batch_size=augument_size,
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train,x_argumented)) # 괄호가 두개인 이유 알아보기! (concatenate는 괄호 두개)
y_train = np.concatenate((y_train,y_argumented))

xy_train=train_datagen2.flow(x_train,y_train, # 형식상 x, y 둘 다 넣어야 함
                                batch_size=64,
                                shuffle=False)  

# print(x_train.shape)
# print(x_test.shape)
# print(x_argumented.shape) #(40000, 28, 28, 1)
# print(x_argumented)        
# print(x_argumented.shape) #(40000, 28, 28, 1)
print(y_train.shape) #(100000,)
print(y_test.shape) #(10000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(100000, 10)
print(y_test.shape) #(10000, 10)


# 2. 모델 구성
model=Sequential()
model.add(Conv2D(32,(2,2), input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(10,activation='softmax'))

# 3. 컴파일, 훈련       다중분류에서 원핫없이 값을 뽑을 수 있다
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
star_time=time.time()
hist=model.fit_generator(xy_train,epochs=1, steps_per_epoch=len(xy_train))

# 4. 평가, 예측
accuracy=hist.history['accuracy']
# val_accuracy=hist.history['val_accuracy']
loss=hist.history['loss']
# val_loss=hist.history['val_loss']
end_time=time.time()-star_time

print('loss:',loss[-1])
# print('val_loss:',val_loss[-1])
print('accuracy:',accuracy[-1])
# print('val_accuracy:',val_accuracy[-1])
print('걸린시간:',end_time)

# loss: 0.9028451442718506
# accuracy: 0.6662999987602234
# 걸린시간: 9.875270128250122
