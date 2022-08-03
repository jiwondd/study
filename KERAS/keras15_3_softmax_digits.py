import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#1. 데이터
datasets=load_digits()
x=datasets['data']
y=datasets.target
# print(x.shape,y.shape) #(1797, 64) (1797,) 인풋 64
# print(np.unique(y)) #[0 1 2 3 4 5 6 7 8 9] 아웃풋 10개 소프트맥스/카테고리컬
print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
print(datasets.feature_names)
import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.imges[1])
# plt.show()
y=to_categorical(y)
print(y)
print(y.shape) #(1797, 10)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)


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
acc score : 1.0


loss: 0.10968619585037231
accuracy: 0.9694444537162781
===================================
acc score : 1.0





'''
