from tabnanny import verbose
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic' #한글폰트 맑은고딕
# matplotlib.rcParams['axes.unicode_minus']=False #유니코드 문제해결
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
model.add(Dense(30,input_dim=13))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

import time 
#3.컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
start_time=time.time()
hist=model.fit(x_train,y_train, validation_split=0.2, 
               epochs=2000, batch_size=100, verbose=1)
end_time=time.time()-start_time
print("걸린시간:",end_time)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
# print('=========================================')
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001B21F4961C0> 
# print('=========================================')
# print(hist.history)
# #{'loss': [548.1727294921875, 128.2350311279297, 103.33795928955078, 104.93355560302734, 98.94366455078125, 83.65609741210938, 81.17222595214844, 74.50555419921875, 75.57810974121094, 69.17296600341797, 62.86097717285156], 'val_loss': [143.34390258789062, 208.70225524902344, 139.1863250732422, 114.55047607421875, 85.66629028320312, 98.64759063720703, 102.41026306152344, 197.17843627929688, 82.43695831298828, 105.87313079833984, 96.37535858154297]}
# # ㄴ딕셔너리 형태 (key,value(list로 되어있음/2개 이상이니까))
# print('=========================================')
# print(hist.history['loss']) #키:밸류 안에서 loss는 문자 그 잡채니까 따옴표 넣어주세여
# print(hist.history['val_loss'])
#히스토리에서 로스만 or 발로스만 빼려면 ->  print(hist.history[loss])
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='*',c='red',label='loss') 
#                            ㄴ .으로 찍어서 보여줘 / 컬러는 레드
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') 
plt.grid() #모눈종이ㄱ
plt.title('loss 와 val_loss') #제목
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()