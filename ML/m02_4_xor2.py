import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

# 1.데이터
x_data=[[0,0],[0,1],[1,0],[1,1]] # ( 4 , 2 )
y_data=[0,1,1,0] #(4, )

# 2.모델구성
# model=LinearSVC()
# model=Perceptron()
model=SVC()

# 3.훈련
model.fit(x_data,y_data)

# 4. 평가, 예측
y_predict=model.predict(x_data)
print(x_data,'의 예축결과:',y_predict)

result=model.score(x_data,y_data)
print('model.score:',result)

acc=accuracy_score(y_data,y_predict)
print('acc_score: ',acc)

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예축결과: [1 1 1 1]
# model.score: 0.5
# acc_score:  0.5 <-model=LinearSVC()

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예축결과: [0 0 0 0]
# model.score: 0.5
# acc_score:  0.5 <-model=Perceptron()

# 리니어도 퍼셉트론도 쏘어는 안되네요! 이래서 인공지능의 1차 겨울이 11년동안 계속되었답니다. 

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예축결과: [0 1 1 0]
# model.score: 1.0
# acc_score:  1.0 <-SVC