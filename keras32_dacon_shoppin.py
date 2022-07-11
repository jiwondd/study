import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler


#.1 데이터
path='./_data/dacon_shopping/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0)

# print(train_set.shape) #(6255, 12)
# print(test_set.shape) #(180, 11)

# print(train_set.columns)
# print(train_set.info())
# print(train_set.describe())


print(train_set.isnull().sum())
# [8 rows x 10 columns]
# Store              0
# Date               0
# Temperature        0
# Fuel_Price         0
# Promotion1      4153
# Promotion2      4663
# Promotion3      4370
# Promotion4      4436
# Promotion5      4140
# Unemployment       0
# IsHoliday          0
# Weekly_Sales       0
# dtype: int64

# 빈 공간 0으로 채우기
train_set=train_set.fillna(0)
test_set=test_set.fillna(0)

# 필요없는 컬럼 제거하기 
train_set = train_set.drop(columns=['Date','IsHoliday'])
test_set = test_set.drop(columns=['Date','IsHoliday'])


x = train_set.drop(columns=['Weekly_Sales'])
y = train_set[['Weekly_Sales']]

print(x.shape) #(6255, 9)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_set=scaler.transform(test_set)

#2. 모델구성
model=Sequential()
model.add(Dense(64,activation='elu',input_dim=9))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='linear'))
model.add(Dense(64,activation='linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
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
result=pd.read_csv(path+'sample_submission.csv',index_col=0)
result['count']=y_summit
result.to_csv(path+'sample_submission.csv',index=True)

# loss: 261013454848.0
# RMSE 510894.77893273067
