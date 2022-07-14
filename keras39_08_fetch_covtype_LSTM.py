from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Flatten, LSTM
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target
# print(x.shape,y.shape)
# print(np.unique(y,return_counts=True)) 
# y=to_categorical(y)
# print(y) 
# print(x.shape,y.shape) #(581012, 54) (581012, 8) -> 투카테고리얼을 쓰면 

y=pd.get_dummies(y)
# print(x.shape,y.shape) #(581012, 54) (581012, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)
print(x_train.shape, x_test.shape) #(464809, 54) (116203, 54)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train = x_train.reshape(464809, 54, 1)
x_test = x_test.reshape(116203, 54, 1)

# #2. 모델구성
model=Sequential()
model.add(LSTM(units=64,input_shape=(54,1)))
model.add(Dense(128,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation='linear'))
model.add(Dense(7,activation='softmax'))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=10, batch_size=100, verbose=1)



#4. 평가, 예측

result=model.evaluate(x_test,y_test)
print('loss:',result[0])
print('accuracy:',result[1])
print("===================================")
y_predict=model.predict(x_test)
y_predict=tf.argmax(y_test,axis=1)
y_test=tf.argmax(y_test,axis=1)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
print('fatch_covtype_끝났당')

# loss: 0.365480899810791
# accuracy: 0.8473103046417236
# ===================================
# acc score : 1.0

# loss: 0.5280614495277405
# accuracy: 0.7746443748474121
# ===================================
# acc score : 1.0 <-LSTM (튜닝전)