from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical
import sklearn as sk


#1. 데이터
datasets=load_digits()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

# 2. 모델구성
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline 
model=make_pipeline(MinMaxScaler(),RandomForestClassifier())

#3. 컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 


# loss: 0.1365455985069275
# accuracy: 0.949999988079071
# ===================================        
# acc score : 1.0  <-RobustScaler


# LinearSVC 결과:  0.9638888888888889
# LinearSVC score : 0.9638888888888889
# =========================================
# LogisticRegression 결과:  0.9611111111111111
# LogisticRegression_acc score : 0.9611111111111111
# =========================================
# KNeighborsClassifier 결과:  0.9194444444444444
# KNeighborsClassifier_acc score : 0.9194444444444444
# =========================================
# DecisionTreeClassifier 결과:  0.8472222222222222
# DecisionTreeClassifier_acc score : 0.8472222222222222
# =========================================
# RandomForestClassifier 결과:  0.9722222222222222
# RandomForestClassifier_acc score : 0.9722222222222222

# model.score: 0.9777777777777777 <-pipeline