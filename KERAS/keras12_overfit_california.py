from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic' #한글폰트 맑은고딕
matplotlib.rcParams['axes.unicode_minus']=False #유니코드 문제해결
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
        train_size=0.8,shuffle=True, random_state=32)

#2. 모델구성
model=Sequential()
model.add(Dense(24,input_dim=8))
model.add(Dense(48))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, epochs=300, batch_size=100,validation_split=0.2)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') 
#                            ㄴ .으로 찍어서 보여줘 / 컬러는 레드
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') 
plt.grid() #모눈종이ㄱ
plt.title('loss 와 val_loss') #제목
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()
