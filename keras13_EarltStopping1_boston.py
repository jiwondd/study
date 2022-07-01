from tabnanny import verbose
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
matplotlib.rcParams['font.family']='Malgun Gothic' #한글폰트 맑은고딕
matplotlib.rcParams['axes.unicode_minus']=False #유니코드 문제해결
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
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(1))

import time 
#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=10,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer="adam")
start_time=time.time()
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)

end_time=time.time()-start_time
print("걸린시간:",end_time)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
print('=========================================')
print(hist.history['val_loss'])

y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)

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


# r2스코어: 0.5686383731151612
# 최소의 val_loss : 59.52528381347656

# loss: 2790.426025390625
# r2스코어: 0.5438157866989157