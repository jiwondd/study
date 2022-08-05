import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

# 1.데이터
x_data=[[0,0],[0,1],[1,0],[1,1]] # ( 4 , 2 )
y_data=[0,1,1,1] #(4, )

# 2.모델구성
# model=LinearSVC()
model=Perceptron()

# 3.훈련
model.fit(x_data,y_data)

# 4. 평가, 예측
y_predict=model.predict(x_data)
print(x_data,'의 예축결과:',y_predict)

result=model.score(x_data,y_data)
print('model.score:',result)

acc=accuracy_score(y_data,y_predict)
print('acc_score: ',acc)

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예축결과: [0 1 1 1]
# model.score: 1.0
# acc_score:  1.0 <-model=LinearSVC()

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예축결과: [0 1 1 1]
# model.score: 1.0
# acc_score:  1.0 <-model=Perceptron()