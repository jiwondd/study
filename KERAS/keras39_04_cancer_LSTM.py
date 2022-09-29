from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D ,Dropout, Flatten, LSTM
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=load_breast_cancer()
# print(datasets.feature_names)
# print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target

# print(x.shape,y.shape) #(569, 30) (569,)


x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

print(x_train.shape,x_test.shape) #(455, 30) (114, 30)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train=x_train.reshape(455, 30, 1)
x_test=x_test.reshape(114, 30, 1)

#2. 모델구성
model=Sequential()                                                   
model.add(LSTM(units=64,input_shape=(30,1)))
model.add(Dense(128,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test)
print("loss : ",loss)
y_predict=model.predict(x_test)

y_predict[(y_predict<0.5)]=0  
y_predict[(y_predict>=0.5)]=1  
acc = accuracy_score(y_test, y_predict)
print('acc score :', acc)
print('cancer_끝났당')

# loss :  0.10539650171995163
# acc score : 0.956140350877193

# loss :  0.14778609573841095
# acc score : 0.9298245614035088 <-LSTM
