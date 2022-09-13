# [실습] 딥하게 구성하기

import numpy as np
from keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()
#인풋 아웃풋이 같다, 특성이 가장 두드러진 부분이 뽑아져나온다. (얼굴을 넣으면 눈코입이 특성이니까 나오고 주근깨 여드름은 지워짐 좋은데?)

x_train=x_train.reshape(60000, 784).astype('float32')/255.
x_test=x_test.reshape(10000, 784).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model=Sequential()
    model.add(Dense(units=hidden_layer_size,input_shape=(784,),
                    activation='relu'))
    model.add(Dense(units=784,activation='sigmoid'))
    return model

# 01번이랑 똑같은데 "함수"형으로 만든거임
# model=autoencoder(hidden_layer_size=64)
model=autoencoder(hidden_layer_size=154) #PCA의 95%성능
model=autoencoder(hidden_layer_size=331) #PCA의 99%성능

model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(x_train,x_train,epochs=1)
output=model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10))=plt.subplots(2,5,figsize=(20,7))
    
random_images=random.sample(range(output.shape[0]),5)

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()

