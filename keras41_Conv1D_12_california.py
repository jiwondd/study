from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten,MaxPooling2D,Dropout, LSTM
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
# print(x)
# print(y)
print(x.shape,y.shape) #(20640, 8) (20640,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape,x_test.shape) #(16512, 8) (4128, 8)

x_train=x_train.reshape(16512, 8, 1)
x_test=x_test.reshape(4128, 8, 1)


#2. 모델구성
model=Sequential()
model.add(Conv1D(10,2,input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('캘리포니아_끝')

# loss: 0.5418692231178284
# r2스코어: 0.5864883952567645

# loss: 0.39999139308929443
# r2스코어: 0.6947584211059534 <-LSTM 으로 바꾼거

# loss: 0.35912272334098816
# r2스코어: 0.7259460778368322 <-Conv1D

