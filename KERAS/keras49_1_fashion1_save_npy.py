# 증폭해서 npy 저장

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

augument_size=40000             # 0~60000(59999) 사이의 값 ㄱ
randidx=np.random.randint(x_train.shape[0],size=augument_size) # ( 60000 , 40000 )
print(x_train.shape) #(60000, 28, 28)
# print(x_train[0].shape) #(28, 28)
# print(x_train[1].shape) #(28, 28)
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

print(x_train.shape)
print(x_test.shape)
print(x_argumented.shape)

x_argumented=train_datagen.flow(x_argumented,y_argumented, # 형식상 x, y 둘 다 넣어야 함
                                batch_size=augument_size,
                                shuffle=False).next()[0]  # .next()[0]을 넣어서 x값만 저장한다. 
print(x_argumented)            # ㄴ위에서 randidx로 이미 랜덤하게 섞어둠
print(x_argumented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train,x_argumented)) # 괄호가 두개인 이유 알아보기! (concatenate는 괄호 두개)
y_train = np.concatenate((y_train,y_argumented))
print(x_train.shape) #(100000, 28, 28, 1)
print(y_train.shape) #(100000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,y_train.shape) #(100000, 28, 28, 1) (100000, 10)
print(x_test.shape,y_test.shape) #(10000, 28, 28, 1) (10000, 10)


np.save('d:/study_data/_save/_npy/keras49_1_train_x.npy',arr=x_train)
np.save('d:/study_data/_save/_npy/keras49_1_train_y.npy',arr=y_train)
np.save('d:/study_data/_save/_npy/keras49_1_test_x.npy',arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_1_test_y.npy',arr=y_test)

