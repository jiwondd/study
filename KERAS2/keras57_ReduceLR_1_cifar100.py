from gc import callbacks
import numpy as np
from keras.datasets import mnist, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000,32*32*3).astype('float32')
x_test = x_test.reshape(10000,32*32*3).astype('float32')

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train = x_train.reshape(50000,32,32,3).astype('float32')
x_test = x_test.reshape(10000,32,32,3).astype('float32')

#2. 모델구성
activation='relu'
optimizer='adam'
drop=0.2

inputs = Input(shape=(32,32,3), name='input')
x = Conv2D(64, (2,2),padding='valid',
           activation=activation, name='hidden1')(inputs)
x = Dropout(drop)(x)
# x = Conv2D(64, (2,2),padding='same',
#            activation=activation, name='hidden2')(inputs)
# x = Dropout(drop)(x)
# x = MaxPooling2D()(x)
# x = Conv2D(32, (3,3),padding='valid',
#            activation=activation, name='hidden3')(inputs)
# x = Dropout(drop)(x)
# x = Flatten()(x) #(25*25*32)
x = GlobalAveragePooling2D()(x)

x = Dense(100, activation=activation, name='hidden4')(x) #20000*256 <-연상양이 너무 많아져서 터진다
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizer, metrics=['acc'],
                loss='sparse_categorical_crossentropy')
    
    
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es=EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auteo',verbose=1,
                            factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=200, validation_split=0.4,batch_size=128,callbacks=[es,reduce_lr])
end = time.time()

from sklearn.metrics import accuracy_score
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('걸린시간 : ', end - start)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))

# 걸린시간 :  210.24202227592468
# loss :  3.5064
# accuracy :  0.1668