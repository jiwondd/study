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
# print(model.layers)
# [<keras.engine.functional.Functional object at 0x0000020EC02A98B0>, <keras.layers.reshaping.flatten.Flatten object at 0x0000020EC6575700>, <keras.layers.core.dense.Dense object at 0x0000020EC65891F0>, <keras.layers.core.dense.Dense object at 0x0000020EC657CD60>]

# 1. include_top=False (모델,가중치 커스터마이징 가능)
# 2. include_top=False + vgg16.trainable=False (가중치동결)
# 3. include_top=False + vgg16.trainable=True + model.trainable=False
print(len(model.weights))           # 1.30 / 2. 30 / 3. 30
print(len(model.trainable_weights)) # 1.30 / 2. 4 / 3. 0
# 가중치까지 가져와서 학습을 시킬건지, 모델만 끌어다가 학습을 시킬건지 고를 수 있다.

import pandas as pd
pd.set_option('max_colwidth',-1)
layers=[(layer,layer.name,layer.trainable) for layer in model.layers]
results=pd.DataFrame(layers,columns=['Layer Type','Layer Name','Layer Trainable'])
print(results)
#                                                               Layer Type Layer Name  Layer Trainable
# 0  <keras.engine.functional.Functional object at 0x000001320C099280>      vgg16      True
# 1  <keras.layers.reshaping.flatten.Flatten object at 0x000001320C0C3BB0>  flatten    True
# 2  <keras.layers.core.dense.Dense object at 0x000001320E920610>           dense      True
# 3  <keras.layers.core.dense.Dense object at 0x000001320E92D700>           dense_1    True