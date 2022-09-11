import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time

x_train=np.load('d:/study_data/_save/_npy/keras47_1_train_x.npy')
y_train=np.load('d:/study_data/_save/_npy/keras47_1_train_y.npy')
x_test=np.load('d:/study_data/_save/_npy/keras47_1_test_x.npy')
y_test=np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')

# print(x_train.shape)    #(100, 200, 200, 3)
# print(y_train.shape)    #(100,)
# print(x_test.shape)     #(100, 200, 200, 3)
# print(y_test.shape)     #(100,)

vgg19=VGG19(weights='imagenet',include_top=False,input_shape=(200,200,3))
vgg19.trainable=False
model=Sequential()
model.add(vgg19)
model.add(Conv2D(1024,(3,3),activation='relu'))
model.add(Conv2D(512,(2,2)))
model.add(Conv2D(256,(2,2),activation='relu'))
model.add(Conv2D(256,(2,2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(128,activation='relu'))
model.add(Dense(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))
model.trainable=False

learnig_rate=0.01
optimizer=Adam(lr=learnig_rate)

model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=300, mode='min', verbose=1, 
                              restore_best_weights=True) 
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, 
                 validation_split=0.2, callbacks=[earlyStopping], verbose=1, batch_size=32)
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)   

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('accuracy : ', acc)
# print('predict:',y_predict)

# loss :  [0.6888511776924133, 0.5099999904632568]
# accuracy :  0.51