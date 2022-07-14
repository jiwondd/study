from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler

#1. 데이터
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print(x_train.shape,y_train.shape) 
print(x_test.shape,y_test.shape)  

x_train=x_train.reshape(60000, 28, 28, 1)
x_test=x_test.reshape(10000, 28, 28, 1)

# print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train=x_train.reshape(60000, -1)
x_test=x_test.reshape(10000, -1)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape,y_train.shape) #(48000, 784) (48000, 10)
print(x_test.shape,y_test.shape) #(12000, 784) (12000, 10)

x_train=x_train.reshape(48000,784,1)
x_test=x_test.reshape(12000,784,1)

#2. 모델구성
model=Sequential()
model.add(LSTM(units=64,input_shape=(784,1)))
model.add(Dense(50,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=50,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=10, batch_size=100, verbose=1)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
print('============================')
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
print('fashion_끝났당')

# loss :  0.3290708065032959
# accuracy :  0.8914166688919067
# ============================
# acc score : 1.0

# loss :  1.008908987045288
# accuracy :  0.6101666688919067
# ============================
# acc score : 1.0 <-LSTM
