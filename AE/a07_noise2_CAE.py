# [실습] 딥하게 구성하기

import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten,UpSampling2D,MaxPooling2D
from keras.models import Sequential, Model

(x_train, _), (x_test, _) = mnist.load_data()

x_train=x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.

x_train_noised=x_train+np.random.normal(0,0.1,size=x_train.shape)
x_test_noised=x_test+np.random.normal(0,0.1,size=x_test.shape) 
#여기에서 0.1을 더해주는 애들이 생기면서 1보다 큰 애들이 생겨버리니까 1보다 큰 애들은 빼버리자
x_train_noised=np.clip(x_train_noised,a_min=0,a_max=1)
x_test_noised=np.clip(x_test_noised,a_min=0,a_max=1)


from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model=Sequential()
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',input_shape=(28,28,1)))  
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32,(2,2),padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(1,(2,2),padding='same',activation='sigmoid')) 
    return model

model=autoencoder(hidden_layer_size=154) #PCA의 95%성능
# model=autoencoder(hidden_layer_size=331) #PCA의 99%성능

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit(x_train_noised,x_train,epochs=10,batch_size=128)
output=model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random

fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),
     (ax11,ax12,ax13,ax14,ax15))=plt.subplots(3,5,figsize=(20,7))
    
random_images=random.sample(range(output.shape[0]),5)

# 원본
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈  
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("NOISE",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()

