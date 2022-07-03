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
from sklearn.metrics import r2_score

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
        train_size=0.8,shuffle=True, random_state=42)

#2. 모델구성
model=Sequential()
model.add(Dense(50,activation='relu',input_dim=13))
model.add(Dense(100,activation='relu')) 
model.add(Dense(80,activation='relu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(30,activation='linear'))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer="adam")#,metrics=['accuracy']
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)
#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
# print(hist.history['val_loss'])

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('loss: ',loss)
print('r2스코어:',r2)

'''
loss: 17.545747756958008 
r2스코어: 0.7876257019581974 <- 기존값

loss:  [-308.109375, 0.0]
r2스코어: -1.6525441280118383 <- r2스코어만 추가함 (머선일..?)

loss:  [21.716720581054688, 0.0]
r2스코어: 0.7175528346687654 ->loss function mse로 바꿈;;ㅎ

loss:  20.230974197387695
r2스코어: 0.7368764177803302

loss:  2.5278218345192727e-06
r2스코어: -5.673201669364601 -> loss=categorical_crossentropy / 마지막 액티베이션 소프트맥스

loss:  22.207286834716797
r2스코어: 0.7111725023087851 -> 다시 mse로 바꾸고 마지막 액티베이션 elu

loss:  21.619325637817383
r2스코어: 0.7188195551952831 ->마지막 액티베이션 linear, 첫번째 relu

loss:  23.756986618041992
r2스코어: 0.6910171269824732 -> 노드 수 조정

loss:  21.364803314208984
r2스코어: 0.7221298798719217 -> 다시 노드 조정

loss:  22.976648330688477
r2스코어: 0.7002920884406907 -> 랜덤스테이트 777

loss:  16.85349464416504
r2스코어: 0.770180980297865 -> 랜덤스테이트 42



'''
