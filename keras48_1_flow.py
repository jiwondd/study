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
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size=100

print(x_train[0].shape) #(28, 28)
print(x_train[0].reshape(28*28).shape) #(784,)
print(np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1).shape) #(100, 28, 28, 1) <- 복제인간 100개 만들었다고 생각하자
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape)



x_data=train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1), # x
    np.zeros(augument_size),                                              # y
    batch_size=augument_size,
    shuffle=True,
).next()
#               next() 사용
print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000028E55E84A30>
print(x_data[0]) #batch 사이즈 만큼 나오겠지? (x, y가 모두 포함)
print(x_data[0].shape) #(100, 28, 28, 1)
print(x_data[1].shape) #(100,)

#               next() 미사용
# print(x_data) 
# print(x_data[0]) #batch 사이즈 만큼 나오겠지? (x, y가 모두 포함)
# print(x_data[0][0].shape) #(100, 28, 28, 1)
# print(x_data[0][1].shape) #(100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i],cmap='gray')
    #plt.imshow(x_data[0][0][i],cmap='gray') # <- next 미사용시엔
    # plt.show(x_data[i],cmap='grey')
plt.show()





