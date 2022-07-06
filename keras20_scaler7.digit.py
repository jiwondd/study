from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score


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

# scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
model=Sequential()
model.add(Dense(120,input_dim=64))
model.add(Dense(200,activation='relu'))
model.add(Dense(160,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='linear'))
model.add(Dense(10,activation='softmax'))

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


'''