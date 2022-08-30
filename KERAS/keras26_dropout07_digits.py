from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input,Dropout
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
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
model=Sequential()
model.add(Dense(120,input_dim=64))
model.add(Dropout(0.3))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(160,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='linear'))
model.add(Dense(10,activation='softmax'))


#3.컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date) #2022-07-07 17:50:42.752072
date=date.strftime('%m%d_%H%M')
print(date) #0707_1750

filepath='./_k24/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'
#        ㄴ4글자로 표시 /로스는 4자리 까지            

earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','digits',filename]))

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
               epochs=100, batch_size=100, verbose=1)

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
loss: 0.14862503111362457
accuracy: 0.9694444537162781
===================================
acc score : 1.0

loss: 0.07681303471326828
accuracy: 0.980555534362793
===================================
acc score : 1.0 <-dropout


'''
