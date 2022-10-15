from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

x_train=np.load('d:/study_data/_save/_npy/keras49_2_train_x.npy')
y_train=np.load('d:/study_data/_save/_npy/keras49_2_train_y.npy')
x_test=np.load('d:/study_data/_save/_npy/keras49_2_test_x.npy')
y_test=np.load('d:/study_data/_save/_npy/keras49_2_test_y.npy')

print(x_train.shape,y_train.shape) #(100000, 28, 28, 1) (100000, 10)
print(x_test.shape,y_test.shape) #(10000, 28, 28, 1) (10000, 10)


#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=10,kernel_size=(4,4),
                 padding='same',
                 input_shape=(28,28,1)))  ##(batch_size, row, column, channels)   
model.add(MaxPooling2D()) #맥스풀링 "레이어"는 아니고 위에 붙어있는 놈이다. 
model.add(Conv2D(8,(3,3),
                 padding='valid', 
                 activation='relu'))
model.add(Conv2D(6,(2,2),
                 padding='valid',
                 activation='relu')) 
model.add(Flatten()) #(N, 63)
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=100, batch_size=10, verbose=1)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  0.23986844718456268
# accuracy :  0.9788333177566528
