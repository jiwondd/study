from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical


#1. 데이터
datasets=load_digits()
x=datasets['data']
y=datasets.target

# print(datasets.feature_names)

y=to_categorical(y)
# print(y)
# print(y.shape) #(1797, 10)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(120,input_dim=64))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(160,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(80,activation='linear'))
# model.add(Dense(10,activation='softmax'))

input1=Input(shape=(64,))
dense1=Dense(120)(input1)
dense2=Dense(200,activation='relu')(dense1)
dense3=Dense(160,activation='relu')(dense2)
dense4=Dense(100,activation='relu')(dense3)
dense5=Dense(80,activation='linear')(dense4)
output1=Dense(10,activation='softmax')(dense5)
model=Model(inputs=input1,outputs=output1)

#3. 컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=50,mode='auto',verbose=1,restore_best_weights=True) #
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=500, batch_size=100, verbose=1)


#4. 평가, 예측

result=model.evaluate(x_test,y_test)
print('loss:',result[0])
print('accuracy:',result[1])
print("===================================")
y_predict=model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)

'''
loss: 0.6078386306762695
accuracy: 0.824999988079071
===================================
acc score : 1.0 <-기존

loss: 0.1150643527507782
accuracy: 0.9666666388511658
===================================
acc score : 1.0 <-MinMax

loss: 0.08361613005399704
accuracy: 0.9722222089767456
===================================
acc score : 1.0 <-Standard

loss: 0.10376747697591782
accuracy: 0.9694444537162781
===================================        
acc score : 1.0   <-MaxAbsScaler
   
loss: 0.1365455985069275
accuracy: 0.949999988079071
===================================        
acc score : 1.0  <-RobustScaler

loss: 0.13214538991451263
accuracy: 0.9638888835906982
===================================
acc score : 1.0 <-함수

'''
