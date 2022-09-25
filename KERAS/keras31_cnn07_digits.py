from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Flatten
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

print(x_train.shape, x_test.shape) #(1437, 64) (360, 64)

x_train = x_train.reshape(1437, 8, 8, 1)
x_test = x_test.reshape(360, 8, 8, 1) 


#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(1,1),input_shape=(8,8,1)))
model.add(Conv2D(128,kernel_size=(1,1),activation='relu'))
model.add(Conv2D(64,kernel_size=(1,1),activation='relu')) 
model.add(Flatten())        
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(42,activation='relu'))
model.add(Dense(18,activation='linear'))
model.add(Dense(10,activation='softmax'))


#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
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


# loss: 0.14862503111362457
# accuracy: 0.9694444537162781
# ===================================
# acc score : 1.0

# loss: 0.07681303471326828
# accuracy: 0.980555534362793
# ===================================
# acc score : 1.0 <-dropout

# loss: 0.1100933626294136
# accuracy: 0.9694444537162781
# ===================================
# acc score : 1.0
