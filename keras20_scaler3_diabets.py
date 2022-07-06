from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=72)

# scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
# print(np.min(x_train)) #0.0
# print(np.max(x_train)) #1.0000000000000004
# print(np.min(x_test)) #0.0
# print(np.max(x_test)) #1.0



#2. 모델구성
model=Sequential()
model.add(Dense(50,activation='relu',input_dim=10))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='linear'))
model.add(Dense(50,activation='linear'))
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
loss: 1080.822998046875
r2스코어: 0.746747006658822 <-기존

loss: 2319.88671875
r2스코어: 0.6486733527571984 <-MinMax

loss: 2764.81640625
r2스코어: 0.5812925903625359 <-Standard








'''