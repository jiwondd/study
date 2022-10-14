import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.covariance import EllipticEnvelope

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!

# print(train_set.isnull().sum())
# print(train_set.info())
# print(train_set.shape) #(1459, 10)
# print(test_set.shape) #(715, 9)

train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
y=train_set['count']
print(x.shape) #(1459, 9)
print(y.shape) #(1459, 9)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)


# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor

model=XGBRegressor()


#3. 컴파일, 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
results=model.score(x_test,y_test)
print('결과:',results)

# 결과: 0.854683534348966