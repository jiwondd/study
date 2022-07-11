from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Flatten
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target
# print(x.shape,y.shape) #(178, 13) (178,)
# print(np.unique(y)) #[0 1 2]
# print(np.unique(y,return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print('=================================')
# print(datasets.DESCR)
# print('=================================')
# print(datasets.feature_names)
y=to_categorical(y)
print(x.shape,y.shape) #(178, 13) (178, 3)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

# scaler=MinMaxScaler()
scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(142, 13) (36, 13)

x_train = x_train.reshape(142, 13, 1, 1)
x_test = x_test.reshape(36, 13, 1, 1) 


# #2. 모델구성
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(1,1),input_shape=(13,1,1)))
model.add(Conv2D(128,kernel_size=(1,1),activation='relu'))
model.add(Conv2D(64,kernel_size=(1,1),activation='relu')) 
model.add(Flatten())        
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(42,activation='relu'))
model.add(Dense(18,activation='linear'))
model.add(Dense(3,activation='softmax'))

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
# print(y_predict)
#[1 2 1 1 0 1 1 2 1 2 1 2 1 2 0 2 1 0 0 0 1 1 1 1 1 1 0 0 0 2 0 1 1 2 1 2]
y_test=np.argmax(y_test,axis=1)
# print(y_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)


'''
# loss: 0.09157252311706543
# accuracy: 0.9722222089767456
# ===================================
# acc score : 1.0

# loss: 0.23664593696594238
# accuracy: 0.8888888955116272
# ===================================
# acc score : 1.0 <-dropout적용

loss: 0.02762087807059288
accuracy: 0.9722222089767456
===================================
acc score : 1.0

'''

