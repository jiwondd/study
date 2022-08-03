from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic' #한글폰트 맑은고딕
matplotlib.rcParams['axes.unicode_minus']=False #유니코드 문제해결
datasets=load_diabetes()
x=datasets.data
y=datasets.target

# print(x)
# print(y)
# print(x.shape,y.shape) #(442, 10) (442,)

# print(datasets.feature_names)
# print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.9,shuffle=True, random_state=42)

#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=10))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, epochs=500, batch_size=10,validation_split=0.3)

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
