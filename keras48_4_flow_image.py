from calendar import c
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

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
test_datagen=ImageDataGenerator(
    rescale=1./255,
)

augument_size=10             # 0~60000(59999) 사이의 값 ㄱ
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

x_train1 = x_train[randidx].copy()
x_train1 = x_train1.reshape(10, 28, 28, 1)

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_argumented=x_argumented.reshape(x_argumented.shape[0],x_argumented.shape[1],x_argumented.shape[2],1)

print(x_train.shape)
print(x_test.shape)
print(x_argumented.shape)

x_argumented=train_datagen.flow(x_argumented,y_argumented,
                                batch_size=augument_size,
                                shuffle=False).next()[0]  # 
print(x_argumented)            # ㄴ위에서 randidx로 이미 랜덤하게 섞어둠
print(x_argumented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train,x_argumented)) # 괄호가 두개인 이유 알아보기! (concatenate는 괄호 두개)
y_train = np.concatenate((y_train,y_argumented))
print(x_train.shape) #(100000, 28, 28, 1)
print(y_train.shape) #(100000,)


fig=plt.figure()
subplot1=fig.add_subplot(1,10,1)
subplot1.plot(x_train,x_argumented)



plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(x_train[0][i],cmap='gray')
    plt.imshow(x_argumented[0][i],cmap='gray')
    # plt.imshow(x_data[0][0][i],cmap='gray') # <- next 미사용시엔
    # plt.show(x_data[i],cmap='grey')
    
plt.show()





