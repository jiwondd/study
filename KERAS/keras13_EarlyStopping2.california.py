from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import time
from tensorflow.python.keras.callbacks import EarlyStopping
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
model.add(Dense(36,input_dim=8))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer="adam")
start_time=time.time()
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=2000, batch_size=100, verbose=1)

end_time=time.time()-start_time
print("걸린시간:",end_time)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)

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

# loss: 0.7869918346405029
# r2스코어: 0.42442032897447823

# loss: 0.7352896928787231
# r2스코어: 0.4622337268768614

# loss: 0.9871588945388794
# r2스코어: 0.2780251865145381
