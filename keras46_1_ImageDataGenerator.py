import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# 1 -> 정상 / 0 -> 이상있음
train_datagen=ImageDataGenerator(
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

test_datagen=ImageDataGenerator(
    rescale=1.255           #평가데이터는 손대지 않고 그 데이터 그대로 해야한다.
)
#                       폴더에서 불러오기ㄱ
xy_train=train_datagen.flow_from_directory(
    'd:/_data/image/brain/train/',
    target_size=(150,150),  #지정사이즈 -> 이미지 수집할때 각자 다 다른 사이즈의 이미지를 임의로 사이즈를 맞춘다
    batch_size=5,
    class_mode='binary',    #test에 ad/normal 두가지 분류니까 만약 3가지 이상이면 categorical
    color_mode='grayscale', #디폴트가 컬러이기 때문에 얘를 빼면 컬러데이터로 나옴
    shuffle=True,
    )                    # Found 160 images belonging to 2 classes.

xy_test=test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary', #test에 ad/normal 두가지 분류니까 만약 3가지 이상이면 categorical
    shuffle=True,
    )                    #Found 120 images belonging to 2 classes.
print(xy_train) #<keras.preprocessing.image.DirectoryIterator object at 0x000001C06A92FD60> 

# from sklearn.datasets import load_boston
# datasets=load_boston()
# print(datasets)

# print(xy_train[0]) # y=[0., 0., 1., 1., 1.]
# print(xy_train[32]) #Asked to retrieve element 32, but the Sequence has length 32 (0~32니까 31이 마지막)
print(xy_train[0][0])  #(5, 150, 150, 3) (5장, 타겟, 타겟, 컬러) (<-흑백도 컬러데이터니까 )
print(xy_train[0][1])
# print(xy_train[31][2].shape) <-error!! y는 0,1 밖에 없잖아요
print(xy_train[31][0].shape,xy_train[31][1].shape) #(5, 200, 200, 1) (5,)

print(type(xy_train))
print(type(xy_train[0]))        #<class 'tuple'>
print(type(xy_train[0][0]))     #<class 'numpy.ndarray'>     /  numpy 두개가 튜플로 묶여있다. 