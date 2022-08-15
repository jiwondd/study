from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical


#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

# print(train_set)
# print(train_set.shape) #(1459, 10)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
# print(test_set)
# print(test_set.shape) #(715, 9)

# print(train_set.columns)
# print(train_set.info())
# print(train_set.describe())

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
        train_size=0.8,shuffle=True, random_state=750)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_set=scaler.transform(test_set)

#2. 모델구성
model=Sequential()
model.add(Dense(45,activation='relu',input_dim=9))
model.add(Dense(90,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(80,activation='linear'))
model.add(Dense(50,activation='linear'))
model.add(Dense(1))

#3.컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date) #2022-07-07 17:50:42.752072
date=date.strftime('%m%d_%H%M')
print(date) #0707_1750

filepath='./_k24/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'
#        ㄴ4글자로 표시 /로스는 4자리 까지            

earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','ddarung',filename]))

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
               epochs=100, batch_size=100, verbose=1)


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


'''
loss: 1799.78955078125
RMSE 42.42392692146283

'''
