from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNetRS101, ResNet152, ResNet152V2
# from keras.applications import ResNetRS101V2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model_list = [VGG16,VGG19,ResNet50,ResNet50V2,ResNetRS101, ResNet152, ResNet152V2,DenseNet121,DenseNet169,DenseNet201,
              InceptionV3,InceptionResNetV2,MobileNet,MobileNetV2,MobileNetV3Small,MobileNetV3Large,
              NASNetLarge,NASNetMobile,EfficientNetB0,EfficientNetB1,EfficientNetB7,Xception]

for model in model_list:
    model = model()
    model.trainable=False
    # model.summary()

    print('==========================================================')
    print(f"모델명 : {model.name}")
    print(f"전체 가중치 갯수 : {len(model.weights)}")
    print(f"훈련가능 가중치 갯수 : {len(model.trainable_weights)}")


'''
==========================================================
모델명 : vgg16
전체 가중치 갯수 : 32
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : vgg19
전체 가중치 갯수 : 38
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : resnet50
전체 가중치 갯수 : 320
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : resnet50v2
전체 가중치 갯수 : 272
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : resnet152v2
전체 가중치 갯수 : 816
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : densenet121
전체 가중치 갯수 : 606
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : densenet169
전체 가중치 갯수 : 846
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : densenet201
전체 가중치 갯수 : 1006
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : inception_v3
전체 가중치 갯수 : 378
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : inception_resnet_v2
전체 가중치 갯수 : 898
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : mobilenet_1.00_224
전체 가중치 갯수 : 137
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : mobilenetv2_1.00_224
전체 가중치 갯수 : 262
훈련가능 가중치 갯수 : 0
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
==========================================================
모델명 : MobilenetV3small
전체 가중치 갯수 : 210
훈련가능 가중치 갯수 : 0
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
==========================================================
모델명 : MobilenetV3large
전체 가중치 갯수 : 266
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : NASNet
전체 가중치 갯수 : 1546
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : NASNet
전체 가중치 갯수 : 1126
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : efficientnetb0
전체 가중치 갯수 : 314
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : efficientnetb1
전체 가중치 갯수 : 442
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : efficientnetb7
전체 가중치 갯수 : 1040
훈련가능 가중치 갯수 : 0
==========================================================
모델명 : xception
전체 가중치 갯수 : 236
훈련가능 가중치 갯수 : 0






'''