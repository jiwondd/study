from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, Dropout,Input
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
# model=Sequential()
# model.add(Conv2D(filters=112,kernel_size=(3,3),
#                  padding='same',
#                  input_shape=(32,32,3))) #컬러라서 맨 뒤가 3
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(100,(3,3),
#                  padding='valid', 
#                  activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.25))
# model.add(Conv2D(88,(2,2),
#                  padding='valid',
#                  activation='relu')) 
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.25))
# model.add(Conv2D(76,(2,2),
#                  padding='valid',
#                  activation='relu')) 
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(132,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100,activation='softmax'))

input1=Input(shape=(32,32,3))
Conv2D1=Conv2D(filters=32, kernel_size=(3,3),activation='relu')(input1)
# Conv2D2=Conv2D(MaxPooling2D())(Conv2D1)
Conv2D2=Conv2D(filters=64, kernel_size=(2,2),activation='relu')(Conv2D1)
Conv2D3=Conv2D(filters=128, kernel_size=(2,2),activation='relu')(Conv2D2)
Conv2D4=Conv2D(filters=64, kernel_size=(2,2),activation='relu')(Conv2D3)
dense1= Flatten()(Conv2D4)
dense2= Dense(32,activation='relu')(dense1)
dense3= Dense(32,activation='relu')(dense2)
output1=Dense(100,activation='softmax')(dense3)
model=Model(inputs=input1,outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=10, batch_size=100, verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
print('============================')
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)

# loss :  3.6480600833892822
# accuracy :  0.1534000039100647
# ============================
# acc score : 1.0

# loss :  4.895394802093506
# accuracy :  0.11819999665021896
# ============================
# acc score : 1.0
