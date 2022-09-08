import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

vgg16=VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))
vgg16.trainable=False
model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10,activation='softmax'))
model.trainable=False

learnig_rate=0.01
optimizer=Adam(lr=learnig_rate)

model.compile(optimizer=optimizer, metrics=['acc'],
                loss='sparse_categorical_crossentropy')

es=EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auteo',verbose=1,
                            factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4,batch_size=128,callbacks=[es,reduce_lr])
end = time.time()

loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print('걸린시간 : ', end - start)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))


# 걸린시간 :  300.0685305595398
# loss :  2.3157
# accuracy :  0.1

# vgg16.trainable=False
# 걸린시간 :  482.0112292766571
# loss :  1.1863
# accuracy :  0.6042

# vgg16.trainable=False + model.trainable=False
# 걸린시간 :  103.2718358039856
# loss :  44.7542
# accuracy :  0.08