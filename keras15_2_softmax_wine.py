import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target
# print(x.shape,y.shape) #(178, 13) (178,)
# print(np.unique(y)) #[0 1 2]
print(np.unique(y,return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print('=================================')
# print(datasets.DESCR)
# print('=================================')
# print(datasets.feature_names)
y=to_categorical(y)
print(y)
print(y.shape) #(178, 3)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)


#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=13))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='linear'))
model.add(Dense(3,activation='softmax'))


#3. 컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=50,mode='auto',verbose=1,restore_best_weights=True) #
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)


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
# loss: 3.036193609237671
# accuracy: 0.5555555820465088
# ===================================
# acc score : 1.0





'''
