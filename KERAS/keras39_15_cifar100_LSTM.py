from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler

#1. 데이터
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
print(x_train.shape,y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

x_train=x_train.reshape(50000, 32, 32, 3)
x_test=x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train=x_train.reshape(50000,-3)
x_test=x_test.reshape(10000,-3)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape,y_train.shape) #(40000, 3072) (40000, 100)
print(x_test.shape,y_test.shape) #(10000, 3072) (10000, 100)

x_train=x_train.reshape(40000,3072,1)
x_test=x_test.reshape(10000,3072,1)



#2. 모델구성
model=Sequential()
model.add(LSTM(units=10,input_shape=(3702,1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(100,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
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
print('cifar100_끝났당')
# loss :  3.3069748878479004
# accuracy :  0.20229999721050262
# ============================
# acc score : 1.0

# loss :  4.497024059295654
# accuracy :  0.02329999953508377
# ============================
# acc score : 1.0 <-LSTM
