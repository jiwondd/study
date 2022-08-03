from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
# print(x)
# print(y)
# print(x.shape,y.shape) #(20640, 8) (20640,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

#2. 모델구성
model=Sequential()
model.add(Dense(40,activation='elu',input_dim=8))
model.add(Dense(80,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(30,activation='linear'))
model.add(Dense(15,activation='linear'))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=3000, batch_size=100, verbose=1)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


'''
loss: 0.43714168667793274
r2스코어: 0.68028926617166

loss: 0.4170139729976654
r2스코어: 0.6764950016026443 <-랜덤스테이트, 액티베이션 변경

loss: 0.746710479259491
r2스코어: 0.4538808915396271 <-...랜덤스테이트 다시 변경ㅎㅎ

loss: 0.43223631381988525
r2스코어: 0.6701515707751251 <-랜덤스테이트 42





'''
