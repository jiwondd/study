import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

# model=VGG16()
model=VGG16(weights='imagenet',include_top=True,input_shape=(224,224,3))
model.summary()
print(len(model.weights)) #include_top=True 32 / include_top=False 26
print(len(model.trainable_weights)) #include_top=True 32 / include_top=False 26
# 우리가 조정할때는 include_top=True로 해주고 인풋 조정해주면된다

################## include_top=True ##################
# 1.FC layer 원래 그대로 쓴다.
# 2.input_shape=(224,224,3) 고정값으로 바꿀 수 없다.
# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ---------------------------------------------------------------
# flatten (Flatten)           (None, 25088)             0
# fc1 (Dense)                 (None, 4096)              102764544
# fc2 (Dense)                 (None, 4096)              16781312
# predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0


################## include_top=False ##################
# 1.FC layer 원래부분 사라지고 커스터마이징이 가능해진다
# 2.input_shape=(32,31,3) 바꿀 수 있다.
# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# ----------flatten 하단 레이어가 사라졌다!!
# 풀리커넥티드 레이어 하단이 없어져벌임