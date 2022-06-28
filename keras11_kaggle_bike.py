#https://www.kaggle.com/competitions/bike-sharing-demand/submit

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame

#.1 데이터
path='./_data/kaggle_bike/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!


#데이트 타임 연/월/일/시 로 컬럼 나누기
train_set['datetime']=pd.to_datetime(train_set['datetime']) #date time 열을 date time 속성으로 변경
#세부 날짜별 정보를 보기 위해 날짜 데이터를 년도, 월, 일, 시간으로 나눠준다.(분,초는 모든값이 0 이므로 추가하지않는다.)
train_set['year']=train_set['datetime'].dt.year
train_set['month']=train_set['datetime'].dt.month
train_set['day']=train_set['datetime'].dt.day
train_set['hour']=train_set['datetime'].dt.hour

#날짜와 시간에 관련된 피쳐에는 datetime, holiday, workingday,year,month,day,hour 이 있다.
#숫자형으로 나오는 holiday,workingday,month,hour만 쓰고 나머지 제거한다.

train_set.drop(['datetime','day','year'],inplace=True,axis=1) #datetime, day, year 제거하기

#month, hour은 범주형으로 변경해주기
train_set['month']=train_set['month'].astype('category')
train_set['hour']=train_set['hour'].astype('category')

#season과 weather은 범주형 피쳐이다. 두 피쳐 모두 숫자로 표현되어 있으니 문자로 변환해준다.
train_set=pd.get_dummies(train_set,columns=['season','weather'])

#casual과 registered는 test데이터에 존재하지 않기에 삭제한다.
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
#temp와 atemp는 상관관계가 아주 높고 두 피쳐의 의미가 비슷하기 때문에 temp만 사용한다.
train_set.drop('atemp',inplace=True,axis=1) #atemp 지우기

#위처럼 test_set도 적용하기
test_set['datetime']=pd.to_datetime(test_set['datetime'])

test_set['month']=test_set['datetime'].dt.month
test_set['hour']=test_set['datetime'].dt.hour

test_set['month']=test_set['month'].astype('category')
test_set['hour']=test_set['hour'].astype('category')

test_set=pd.get_dummies(test_set,columns=['season','weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y=train_set['count']

# print(train_set.shape) #(10886, 16)
# print(test_set.shape) #(6493, 15)
# print(x.shape) #(10886, 15)
# print(y.shape) #(10886,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=777)

#2. 모델구성
model=Sequential()
model.add(Dense(32,input_dim=15))
model.add(Dense(60,activation='ReLU'))
model.add(Dense(100,activation='ReLU'))
model.add(Dense(50,activation='ReLU'))
model.add(Dense(30,activation='ReLU'))
model.add(Dense(10,activation='ReLU'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train,epochs=600,batch_size=100)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("RMSE",rmse)



y_summit=model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)


result=pd.read_csv(path+'sampleSubmission.csv',index_col=0)
result['count']=y_summit
result=abs(result)
result.to_csv(path+'sampleSubmission.csv',index=True)

# loss: 2912.559326171875
# RMSE 53.968133997201804



'''

train_size=0.99, random_state=750 ,epo=100, batch_size=100
loss: 5565.4990234375
RMSE 74.60226904357341

train_size=0.99, random_state=777 ,epo=300, batch_size=100
loss: 3464.6220703125
RMSE 58.86104163102938
loss: 3355.15673828125
RMSE 57.92371660738908

train_size=0.8, random_state=777 ,epo=300, batch_size=100
loss: 5911.56494140625
RMSE 76.88669869799274

train_size=0.8, random_state=31 ,epo=300, batch_size=100
loss: 5155.96435546875
RMSE 71.80503666686437

train_size=0.8, random_state=21 ,epo=300, batch_size=100
loss: 4914.68505859375
RMSE 70.10482004893267

train_size=0.8, random_state=21 ,epo=300, batch_size=100, activation='elu'
loss: 4844.0966796875
RMSE 69.59955220145058

train_size=0.8, random_state=777 ,epo=300, batch_size=100, activation='elu'
loss: 5443.865234375
RMSE 73.7825521223057

train_size=0.8, random_state=666 ,epo=300, batch_size=100, activation='elu'
loss: 32250.59375
RMSE 179.58450716008647

train_size=0.99, random_state=777 ,epo=300, batch_size=100, activation='elu'
loss: 3705.72705078125
RMSE 60.87468315938493

train_size=0.99, random_state=777 ,epo=300, batch_size=100, activation='elu'
result=abs(result)
loss: 3810.980712890625
RMSE 61.73313440188133

train_size=0.99, random_state=777 ,epo=300, batch_size=100, activation='ReLU'
loss: 3743.863037109375
RMSE 61.18711602308837

train_size=0.99, random_state=777 ,epo=300, batch_size=100, activation='ReLU'
result=abs(result)
loss: 3614.044189453125
RMSE 60.11692504201371

train_size=0.8, random_state=750 ,epo=300, batch_size=100, activation='ReLU'
result=abs(result)
loss: 6140.11279296875
RMSE 78.35888126382463

train_size=0.8, random_state=750 ,epo=300,batch_size=10,verbose=0, activation='ReLU'
result=abs(result)
loss: 7081.57373046875
RMSE 84.15209177202996

train_size=0.99, random_state=777 ,epo=300,batch_size=10,verbose=0, activation='ReLU'
result=abs(result)
loss: 4337.7041015625
RMSE 65.86124750629446

train_size=0.99, random_state=777 ,epo=300,batch_size=100, activation='ReLU'
result=abs(result) +레이어 갯수 늘림
loss: 4342.3505859375
RMSE 65.89651558762826

노드+레이어 조정
train_size=0.99, random_state=777 ,epo=500,batch_size=100, activation='ReLU'
result=abs(result)
loss: 3866.517822265625
RMSE 62.181329019818776   (0.58)

train_size=0.99, random_state=777 ,epo=300,batch_size=100, activation='ReLU'
result=abs(result)
loss: 2912.559326171875
RMSE 53.968133997201804

loss: 3332.6259765625
RMSE 57.72889853127565 / 0.60

loss: 3524.21533203125
RMSE 59.365096505651586 /activation='selu'

'''