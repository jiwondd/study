from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. 데이터
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)
#reshape했을때 서로의 곱한 값이 같아야 한다,순서가 섞이면 안된다. 

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

print(x_train.shape) #(60000, 28, 28, 1)
print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)


#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=10,kernel_size=(4,4),
                 padding='same',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(8,(3,3),
                 padding='valid', 
                 activation='relu'))
model.add(Conv2D(6,(2,2),
                 padding='valid',
                 activation='relu')) 
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=100, batch_size=100, verbose=1)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  4.605644702911377
# accuracy :  0.008700000122189522
# ============================
# acc score : 1.0
