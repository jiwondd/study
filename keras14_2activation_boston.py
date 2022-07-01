# [과제] 
# activation : sigmoid, relu, linear 넣기
# metrics 추가
# Earlystopping 추가
# 성능비교
# 느낀점 2줄 이상 쓰기

from tabnanny import verbose
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

datasets=load_boston()
x=datasets.data
y=datasets.target

#1. 데이터
# print(x)
# print(y)
# print(x.shape, y.shape) #(506, 13) (506,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

#2. 모델구성
model=Sequential()
model.add(Dense(50,input_dim=13))
model.add(Dense(80,activation='relu')) 
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)
#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print(hist.history['val_loss'])

y_predict=model.predict(x_test)
