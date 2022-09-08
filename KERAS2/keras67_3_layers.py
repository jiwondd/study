from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

########## 레이어별로 동결시키기 ##########

# model.trainable=False
# =================================================================
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17
# _________________________________________________________________

# for layer in model.layers:
#     layer.trainable=False
# =================================================================
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17
# _________________________________________________________________

# model.layers[0].trainable=False # dense (Dense)  
# =================================================================
# Total params: 17
# Trainable params: 11
# Non-trainable params: 6
# _________________________________________________________________
# for layer in model.layers[0]:
#     layer.trainable=False

# model.layers[1].trainable=False # dense_1 (Dense)
# =================================================================
# Total params: 17
# Trainable params: 9
# Non-trainable params: 8
# _________________________________________________________________

model.layers[2].trainable=False # dense_2 (Dense) 
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 14
# Non-trainable params: 3
# _________________________________________________________________

model.summary()
