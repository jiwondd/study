from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# #2. 모델구성
# model=Sequential()
# model.add(Dense(60,input_dim=30))
# model.add(Dense(90,activation='relu')) 
# model.add(Dense(50,activation='relu')) 
# model.add(Dense(20,activation='relu'))
# model.add(Dense(10,activation='sigmoid'))
# model.add(Dense(1,activation='sigmoid'))

# #3.컴파일, 훈련
# earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
# model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
# hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
#                epochs=1000, batch_size=100, verbose=1)

# model.save("./_save/keras23_10_save_model_cancer.h5")
model=load_model("./_save/keras23_10_save_model_cancer.h5")

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test)
print("loss : ",loss)
y_predict=model.predict(x_test)

# y_predict[(y_predict<0.5)]=0  
# y_predict[(y_predict>=0.5)]=1  
# acc = accuracy_score(y_test, y_predict)
print('acc score :', acc)



'''
loss :  [0.1798676997423172, 0.9122806787490845]
acc score : 0.8508771929824561 <-기존

loss :  0.1710389256477356
acc score : 0.9561403393745422  <-MinMax

loss :  0.21601618826389313
acc score : 0.9473684430122375  <-Standard

loss :  0.14239634573459625
acc score : 0.9561403393745422  <-MaxAbsScaler

loss :  0.23208273947238922
acc score : 0.9385964870452881 <-RobustScaler

loss :  0.1924625039100647
acc score : 0.9385964870452881 <-함수

loss :  0.14482766389846802
acc score : 0.9561403393745422 <- load_model


'''
