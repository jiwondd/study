from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, Conv1D, Reshape, LSTM, GRU
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



# 2. 모델구성
# model=Sequential()
# model.add(Conv2D(filters=64,kernel_size=(3,3),
#                  padding='same',  #(batch_size, row, column, channels)   
#                  input_shape=(28,28,1)))     #(28, 28, 64)          Param = 640
# model.add(MaxPooling2D())                    #(14 , 14, 64)         Param = 0
# model.add(Conv2D(32,(3,3)))                  #(12, 12, 32)          Param = 18464
# model.add(Conv2D(7,(3,3)))                   #(10, 10, 7)           Param = 2023
# model.add(Flatten())                         #(N, 700)              Param = 0
# model.add(Dense(100,activation='relu'))      #(N, 100)              Param = 70100
# model.add(Reshape(target_shape=(100,1)))     #(N, 100, 1)           Param = 0
# model.add(Conv1D(10, 3))                     #(N, 98, 10)           Param = 40
# model.add(LSTM(16))                          #(N, 16)               Param = 1728
# model.add(Dense(32,activation='relu'))       #(N, 32)               Param = 544
# model.add(Dense(10,activation='softmax'))    #(N, 10)               Param = 3300
# model.summary()

model=Sequential()
model.add(Conv1D(10,2,input_shape=(3,1)))
model.add(Dense(10))
model.add(Reshape(target_shape=(1,2,2,5)))
model.add(Conv2D(filters=10,kernel_size=(1,1),input_shape=(2,2,5)))
model.add(Reshape(target_shape=(8,5)))
model.add(LSTM(8))
model.add(Dense(10))
model.summary()

'''
model=Sequential()
model.add(Conv2D(filters=10,kernel_size=(2,2),input_shape=(20,20,1))) 
# 필터/유닛 = 10 / 커널사이즈 2,2 / N, rows, columns, color/channels (20,20,1)
model.add(Conv2D(filters=8,kernel_size=(2,2))) 
# 필터/유닛 = 8 / 커널사이즈 2,2 / N, rows, columns, color/channels (N,18,18,8)
model.add(Flatten())  
# 3차원의 데이터를 쭉 펼쳐서 2차원으로 만들자 (N, 2592)
model.add(Dense(10,activation='relu'))  
# (N, 10)
model.add(Reshape(target_shape=(10,1))) 
# 쉐입을 바꾸자 (N, 10, 1)
model.add(Conv1D(10, 3))
# 필터/유닛, 커널사이즈 3 (N, 8, 10)
model.add(LSTM(16,return_sequences=True))
# 필터 (N, 8, 8) / 밑에도 같은 사이즈로 보내줘
model.add(GRU(6))
# 필터 (N, 6)
model.summary()


# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=100, batch_size=100, verbose=1)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  0.23986844718456268
# accuracy :  0.9788333177566528
'''