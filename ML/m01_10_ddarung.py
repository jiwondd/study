import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.svm import LinearSVR

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
y=train_set['count']

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)

#2. 모델구성
model=LinearSVR()

#3. 컴파일, 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result=model.score(x_test,y_test)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('결과: ',result)


# loss: 722.6207885742188
# RMSE 26.881613397126884

# r2스코어: 0.7483147676970445
# 결과:  0.7483147676970445 <-LinearSVR

