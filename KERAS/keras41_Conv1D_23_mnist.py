# CNN -> LSTM

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten,MaxPooling2D,Dropout,LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 데이터
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

x_train= x_train.reshape(60000, 28, 28,1)
x_test=x_test.reshape(10000,28,28,1)
# print(x_train.shape) #(60000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train=x_train.reshape(60000,-1)
x_test=x_test.reshape(10000,-1)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)

# print(x_train.shape,y_train.shape) #(48000, 784) (48000, 10)
# print(x_test.shape,y_test.shape) #(12000, 784) (12000, 10)

x_train=x_train.reshape(48000,784,1)
x_test=x_test.reshape(12000,784,1)

print(x_train.shape,y_train.shape) #(48000, 784) (48000, 10)
print(x_test.shape,y_test.shape) #(12000, 784) (12000, 10)


# 2. 모델구성
model=Sequential()
model.add(Conv1D(10,2,input_shape=(784,1)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation='linear'))
model.add(Dense(10,activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=10, batch_size=100, verbose=1)


# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
print('============================')
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
print('mnist_끝났음')



# loss :  0.12688107788562775
# accuracy :  0.9699166417121887
# ============================
# acc score : 1.0

# loss :  1.6741256713867188
# accuracy :  0.3761666715145111
# ============================
# acc score : 1.0                   <-LSTM

# loss :  0.2034139633178711
# accuracy :  0.9440000057220459
# ============================
# acc score : 1.0 <-Conv1D
