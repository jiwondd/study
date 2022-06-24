import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
print(train_set)
print(train_set.shape) #(1459, 10)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
print(test_set)
print(test_set.shape) #(715, 9)

print(train_set.columns)
print(train_set.info())
print(train_set.describe())

###결측치 처리하기 1. 제거하기 ###
print(train_set.isnull().sum()) #널의 갯수를 더해라 /컬럼 당 결측치의 갯수를 확인 할 수 있다.
train_set=train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape) #(1328, 10) 130개 정도 사라졌음ㅎ
#####

x=train_set.drop(['count'],axis=1)
print(x)
print(x.columns)
print(x.shape) #(1459, 9)

y=train_set['count']
print(y)
print(y.shape) #(1459,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.9,shuffle=True, random_state=1004)

#2. 모델구성
model=Sequential()
model.add(Dense(18,input_dim=9))
model.add(Dense(36))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train,epochs=2500,batch_size=10)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("RMSE",rmse)

# loss: 2429.943115234375
# RMSE 49.294453251186695

# y_predict=model.predict(test_set)
#loss: 2886.80322265625
