import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader


#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

print(train_set)
print(train_set.shape) #(1459, 10)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
print(test_set)
print(test_set.shape) #(715, 9)

print(train_set.columns)
print(train_set.info())
print(train_set.describe())

###결측치 처리하기 1. 제거하기 ###
#print(train_set.isnull().sum()) #널의 갯수를 더해라 /컬럼 당 결측치의 갯수를 확인 할 수 있다.
train_set=train_set.dropna()
#print(train_set.isnull().sum())
#print(train_set.shape) #(1328, 10) 130개 정도 사라졌음ㅎ
#####
test_set=test_set.fillna(0)

x=train_set.drop(['count'],axis=1)
# print(x)
# print(x.columns)
# print(x.shape) #(1459, 9)

y=train_set['count']
# print(y)
# print(y.shape) #(1459,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)

#2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=9))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train,epochs=300,batch_size=10)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("RMSE",rmse)

y_summit=model.predict(test_set)
print(y_summit)
print(y_summit.shape)

result=pd.read_csv(path+'submission.csv',index_col=0)
#임의로 인덱스를 맞춤

result['count']=y_summit
#카운트에 y서브밋을 넣고

result.to_csv(path+'submission.csv',index=True)

# loss: 722.6207885742188
# RMSE 26.881613397126884

# df=pd.DataFrame.from_dict(y_summit)
# df.to_csv('C:\study\_data\ddarung\submission.csv',header=True)

'''
train_size=0.99,shuffle=True, random_state=750 / epo 300
loss: 823.4113159179688
RMSE 28.695140264473554

train_size=0.90,shuffle=True, random_state=750 / epo 300
loss: 2097.272705078125
RMSE 45.7959939584777

train_size=0.90,shuffle=True, random_state=777 / epo 300
loss: 2747.86962890625
RMSE 52.420125336383194

train_size=0.90,shuffle=True, random_state=21 / epo 300
loss: 3245.165771484375
RMSE 56.96635699192433

train_size=0.99,shuffle=True, random_state=750 / epo 300
loss: 947.4698486328125
RMSE 30.781000344873252

train_size=0.99,shuffle=True, random_state=750 / epo 3000
loss: 869.1902465820312
RMSE 29.482029375203123

train_size=0.9,shuffle=True, random_state=750 / epo 3000
loss: 2293.812255859375
RMSE 47.8937581648488

train_size=0.9,shuffle=True, random_state=750 / epo 300
loss: 2625.174560546875
RMSE 51.23645884676348

train_size=0.98,shuffle=True, random_state=750 / epo 300
loss: 3694.760498046875
RMSE 60.78454202594039

train_size=0.99,shuffle=True, random_state=750 / epo 300
loss: 1147.1170654296875
RMSE 33.86911689700243

train_size=0.99,shuffle=True, random_state=750 / epo 400
loss: 1430.2646484375
RMSE 37.818841096871296

train_size=0.99,shuffle=True, random_state=750 / epo 4000
loss: 1313.485595703125
RMSE 36.242045086988824



train_size=0.99,shuffle=True, random_state=21 / epo 500
model.add(Dense(40,aviation='selu'))
loss: 636.6697387695312
RMSE 25.232314933987098
실행 할때마다 편차가 너무 커서 신뢰하기 어렵다
'''