import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

# model=VGG16()
# vgg16=VGG16(weights='imagenet',include_top=True,input_shape=(224,224,3))
vgg16=VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))


vgg16.trainable=True #가중치동결=Flase
vgg16.summary()

model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
# model.trainable=False (새로 추가한 노드도 가중치 갱신 안해)
model.summary() 

# 1. include_top=False (모델,가중치 커스터마이징 가능)
# 2. include_top=False + vgg16.trainable=False (가중치동결)
# 3. include_top=False + vgg16.trainable=True + model.trainable=False
print(len(model.weights))           # 1.30 / 2. 30 / 3. 30
print(len(model.trainable_weights)) # 1.30 / 2. 4 / 3. 0

# 가중치까지 가져와서 학습을 시킬건지, 모델만 끌어다가 학습을 시킬건지