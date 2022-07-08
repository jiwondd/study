from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, Dropout
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
print(x_train.shape,y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) #(10000, 32, 32, 3) (10000, 1)

x_train=x_train.reshape(50000, 32, 32, 3)
x_test=x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train, return_counts=True))
'''
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,  
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,  
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,     
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,     
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,     
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,     
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,     
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,     
       500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)

#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=90,kernel_size=(10,10),
                 padding='same',
                 input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(140,(8,8),
                 padding='valid', 
                 activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(80,(6,6),
                 padding='valid',
                 activation='relu')) 
model.add(Dropout(0.25))
model.add(Conv2D(60,(4,4),
                 padding='valid',
                 activation='relu')) 
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=300, batch_size=100, verbose=1)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
print('============================')
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)

# loss :  4.605644702911377
# accuracy :  0.008700000122189522
# ============================
# acc score : 1.0