# 넘파이 불러와서 모델 구성
# 성능비교

from calendar import c
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, LSTM
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

x_train=np.load('d:/study_data/_save/_npy/keras49_1_train_x.npy')
y_train=np.load('d:/study_data/_save/_npy/keras49_1_train_y.npy')
x_test=np.load('d:/study_data/_save/_npy/keras49_1_test_x.npy')
y_test=np.load('d:/study_data/_save/_npy/keras49_1_test_y.npy')

print(x_train.shape,y_train.shape) #(100000, 28, 28, 1) (100000, 10)
print(x_test.shape,y_test.shape) #(10000, 28, 28, 1) (10000, 10)


# 2. 모델 구성
model=Sequential()
model.add(Conv2D(32,(2,2), input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(10,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
star_time=time.time()
hist=model.fit(x_train,y_train,epochs=10, validation_split=0.3)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)
print('loss : ', loss[0])
# print('accuracy : ', loss[1])
# print('============================')
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)

# loss :  0.6381710171699524
# acc score : 0.8917
