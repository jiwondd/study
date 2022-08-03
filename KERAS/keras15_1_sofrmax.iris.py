import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
tf.random.set_seed(777) #<-웨이트값을 랜덤하게 섞는것

#1. 데이터 
datasets=load_iris()
# print(datasets.DESCR)
#    :Number of Instances: 150 (50 in each of three classes) / 150개의 행
#    :Number of Attributes: 4 numeric, predictive attributes and the class /4개의 컬럼, 열, 피쳐, 특성
# print(datasets.feature_names)
x=datasets['data']
y=datasets.target
# print(x)
# print(x.shape) #(150, 4)
# print(y)
# print(y.shape) #(150,)


# tensorflow.kersa에서의 one-hot-encoding
print('y의 라벨값:',np.unique(y)) #   y의 라벨값: [0 1 2]
y=to_categorical(y)
print(y)
print(y.shape) #(150, 3)



x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)


#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=4))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='linear'))
model.add(Dense(3,activation='softmax'))


#3. 컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=50,mode='min',verbose=1,restore_best_weights=True) #
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)

#4. 평가, 예측

# 1.첫번째 방법
# loss,acc=model.evaluate(x_test,y_test)
# print("loss : ",loss)
# print('acc score :', acc)


# 2.두번재 방법
result=model.evaluate(x_test,y_test)
print('loss:',result[0])
print('accuracy:',result[1])
# print("===================================")
# print(y_test[:5])
# print("===================================")
# y_pred=model.predict(x_test[:5])
# print(y_pred)
print("===================================")

y_predict=model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
print(y_predict)
# [1 2 0 1 2 0 2 1 0 0 2 1 2 0 2 1 1 1 2 0 2 2 0 2 1 0 1 1 1 1]
y_test=np.argmax(y_test,axis=1)
print(y_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)



'''
loss :  [0.10553205758333206, 0.6666666865348816]
acc score : 0.6666666666666666 

loss :  -140761890816.0
acc score : 0.4666666666666667

loss: 0.4667608439922333
acc score : 0.9333333333333333

loss: 0.6776479482650757
accuracy: 0.9333333373069763

loss :  0.7089844942092896
acc score : 0.9333333373069763

loss :  0.7647645473480225
acc score : 0.9333333373069763 -> 위쪽에 웨이트값에 대한 랜덤스테이트 추가 한 값

loss: 0.3925118148326874
accuracy: 0.9333333373069763
===================================
[1 2 0 1 2 0 2 1 0 0 2 1 2 0 2 1 1 1 2 0 2 2 0 2 1 0 1 1 1 1]
[1 2 0 1 2 0 2 1 0 0 2 1 2 0 2 1 1 1 2 0 2 2 0 2 1 0 1 1 1 1]     
acc score : 1.0



'''
