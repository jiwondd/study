# 단층 퍼셉트론
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 1.데이터
x_data=[[0,0],[0,1],[1,0],[1,1]] # ( 4 , 2 )
y_data=[0,1,1,0] #(4, )

# 2.모델구성
# model=LinearSVC()
# model=Perceptron()
model=Sequential()
model.add(Dense(1,input_dim=2,activation='sigmoid'))

# 3.훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_data,y_data,batch_size=1,epochs=100)

# 4. 평가, 예측
y_predict=model.predict(x_data)
print(x_data,'의 예축결과:',y_predict)
result=model.evaluate(x_data,y_data)
print('metrics score:',result[1])

# acc=accuracy_score(y_data,y_predict)
# print('acc_score: ',acc)
