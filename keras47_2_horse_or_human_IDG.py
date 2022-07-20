from pickletools import optimize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib
import matplotlib.pyplot as plt
# 1 -> 정상 / 0 -> 이상있음

# 1. 데이터
xy_dataset=ImageDataGenerator(
    rescale=1./255, #MinMaxs
    horizontal_flip=True,   #수평반전
    vertical_flip=True,     #수직반전
    width_shift_range=0.1,  #좌우이동
    height_shift_range=0.1, #상하이동
    rotation_range=5,       #임의의 각도록 회전시킨다.
    zoom_range=1.2,         #확대범위
    shear_range=0.7,        #기울기(?)/찌그러트리면서 돌리는?
    fill_mode='nearest'     #여기까지 다 넣어도 랜덤으로 선택되서 조정됨
)

xy_dataset=xy_dataset.flow_from_directory(
    'd:/study_data/_data/image/horse-or-human/',
    target_size=(200,200),  #지정사이즈 -> 이미지 수집할때 각자 다 다른 사이즈의 이미지를 임의로 사이즈를 맞춘다
    batch_size=100,
    class_mode='binary',    #test에 ad/normal 두가지 분류니까 만약 3가지 이상이면 categorical
    #color_mode='', #디폴트가 컬러이기 때문에 얘를 빼면 컬러데이터로 나옴
    shuffle=True,
    )                   
print(xy_dataset) # Found 1027 images belonging to 2 classes.

# xy_test=test_datagen.flow_from_directory(
#     'd:/study_data/_data/image/horse-or-human/test_set/',
#     target_size=(200,200),
#     batch_size=100,
#     class_mode='binary', 
#     shuffle=True,
#     )                    
# print(xy_test) #Found 2023 images belonging to 2 classes.

# print(xy_dataset[0][0])  
# print(xy_dataset[0][1])
print(xy_dataset[10][0].shape,xy_dataset[0][1].shape) #(27, 200, 200, 3) (100,)


np.save('d:/study_data/_save/_npy/keras47_2_train_x.npy',arr=xy_dataset[0][0])
np.save('d:/study_data/_save/_npy/keras47_2_train_y.npy',arr=xy_dataset[0][1])
# np.save('d:/study_data/_save/_npy/keras47_1_test_x.npy',arr=xy_test[0][0])
# np.save('d:/study_data/_save/_npy/keras47_1_test_y.npy',arr=xy_test[0][1])
