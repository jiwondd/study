import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target
print(x.shape,y.shape) #(569, 30) (569,) imput 30/output 1
print(x)
print(y)

#2. 모델구성
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

model=Sequential()
model.add(Dense(50,input_dim=30))
model.add(Dense(80,activation='sigmoid')) 
model.add(Dense(50,activation='relu')) #렐루는 중간 레이어에서만 가능함 / 정확도 80퍼이상
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
# **이진분류에서 로스는 binary_crossentropy를 사용한다**  
# R2는 회귀모델에서 쓰이니까 분류모델에서는 다른 평가지표를 사용한다 metrics=['accuracy']
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)

# print('loss:',loss)
# print('=========================================')
# print(hist.history['val_loss'])

y_predict=model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score
print(y_predict)

# [과제] accuracy 값을 완성하자
# r2=r2_score(y_test,y_predict)
# acc=accuracy_score(y_test,y_predict)
# print('acc score:',acc)
# ㄴR2는 회귀모델에 쓰이는 평가지표니까 안쓴다!
'''
# loss: 0.11057772487401962
# r2스코어: 0.5024002929908372

# r2스코어: 0.81025255250959 -> 마지막 레이어 시그모이드 적용 후

# loss: [0.1370796114206314, 0.9385964870452881]
#        ㄴ얘는 원래 LOSS  /  ㄴ매트릭스의 값 (평가지표 두개[리스트]니까 두개나온거임)


'''