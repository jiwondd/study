import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from keras.applications.regnet import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D,UpSampling2D,MaxPooling2D
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

x_data=np.load('d:/study_data/_save/_npy/keras47_4_train_x.npy')
# y_data=np.load('d:/study_data/_save/_npy/keras47_4_train_y.npy')
# x_test=np.load('d:/study_data/_save/_npy/keras47_4_test_x.npy')
my_pic=np.load('d:/study_data/_save/_npy/keras47_4_test.npy')
# y_test=np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')

x_train, x_test = train_test_split(x_data, test_size=0.3, shuffle=True, random_state=42)

x_train_noised=x_train+np.random.normal(0,0.1,size=x_train.shape)
x_test_noised=x_test+np.random.normal(0,0.1,size=x_test.shape) 
my_pic_noised=my_pic+np.random.normal(0,0.1,size=my_pic.shape) 

x_train_noised=np.clip(x_train_noised,a_min=0,a_max=1)
x_test_noised=np.clip(x_test_noised,a_min=0,a_max=1)
my_pic_noised=np.clip(my_pic_noised,a_min=0,a_max=1)

# print(x_train.shape)    #(70, 200, 200, 3)
# print(x_test.shape)     #(30, 200, 200, 3)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', strides=2,input_shape=(200,200,3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


# model=autoencoder(hidden_layer_size=154) #PCA의 95%성능
model=autoencoder(hidden_layer_size=64) #PCA의 99%성능
# model.fit(x_train_noised,x_train,epochs=30,batch_size=1024) <-배치사이즈 크면 이미지 난리남;;
model.fit(x_train_noised,x_train,epochs=30,batch_size=16)
output=model.predict(x_test_noised)
my_pred=model.predict(my_pic_noised)

import matplotlib.pyplot as plt
import random

fig,((ax1,ax2,ax3,ax4,ax5,m1),(ax6,ax7,ax8,ax9,ax10,m2),
     (ax11,ax12,ax13,ax14,ax15,m3))=plt.subplots(3,6,figsize=(20,7))
    
random_images=random.sample(range(output.shape[0]),5)

# 원본
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(200,200,3))
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
m1.imshow(my_pic[0].reshape(200,200,3))    
    
# 노이즈  
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(200,200,3))
    if i==0:
        ax.set_ylabel("NOISE",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
m2.imshow(my_pic_noised[0].reshape(200,200,3))    
    
# 
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(200,200,3))
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
m3.imshow(my_pred[0].reshape(200,200,3))    
    
plt.tight_layout()
plt.show()
