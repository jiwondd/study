from keras.models import Model
from keras.layers import Dense, Flatten, Input, Dropout,GlobalAveragePooling2D
from keras.applications import VGG16, InceptionV3
from keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

base_model=InceptionV3(weights='imagenet',include_top=False)
base_model.summary()

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)

output=Dense(100,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=output)

# 1. 
for layer in base_model.layers: #base_model.layers[3] <-이러캐하면 특정레이어만 false
    layer.trainable=False
# ==================================================================================================
# Total params: 24,003,460
# Trainable params: 2,200,676
# Non-trainable params: 21,802,784
# __________________________________________________________________________________________________

# 2.
# base_model.trainable=False
# ==================================================================================================
# Total params: 24,003,460
# Trainable params: 2,200,676
# Non-trainable params: 21,802,784
# __________________________________________________________________________________________________
    
# model.summary()
# print(base_model.layers)