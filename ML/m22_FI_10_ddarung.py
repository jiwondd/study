import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count','hour_bef_precipitation','hour_bef_humidity'],axis=1)
y=train_set['count']

print(x.shape) #(1328, 9)->(1328, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model1=DecisionTreeRegressor()
model2=RandomForestRegressor()
model3=GradientBoostingRegressor()
model4=XGBRegressor()

# 3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4.평가, 예측
result1=model1.score(x_test,y_test)
print(model1,'의 model.score:',result1)
y_predict1=model1.predict(x_test)
r21=r2_score(y_test,y_predict1)
print(model1,'의 r2_score :',r21)
print(model1,':',model1.feature_importances_)
print('*************************************************')
result2=model2.score(x_test,y_test)
print(model2,'의 model.score:',result2)
y_predict2=model2.predict(x_test)
r22=r2_score(y_test,y_predict2)
print(model2,'의 r2_score :',r22)
print(model2,':',model2.feature_importances_)
print('*************************************************')
result3=model3.score(x_test,y_test)
print(model3,'의 model.score:',result3)
y_predict3=model3.predict(x_test)
r23=r2_score(y_test,y_predict3)
print(model3,'의 r2_score :',r23)
print(model3,':',model3.feature_importances_)
print('*************************************************')
result4=model4.score(x_test,y_test)
print(model4,'의 model.score:',result4)
y_predict4=model4.predict(x_test)
r24=r2_score(y_test,y_predict4)
print(model4,'의 r2_score :',r24)
print(model4,':',model4.feature_importances_)
print('*************************************************')

# DecisionTreeRegressor() 의 model.score: 0.8988769296199881
# DecisionTreeRegressor() 의 r2_score : 0.8988769296199881
# DecisionTreeRegressor() : [0.57668716 0.19298587 0.01248489 0.03504973 0.04227141 0.02562034
#  0.03939855 0.05298862 0.02251344]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.9108836404679226
# RandomForestRegressor() 의 r2_score : 0.9108836404679226
# RandomForestRegressor() : [0.58739117 0.1843266  0.01617893 0.03228564 0.03965585 0.03744703
#  0.04055366 0.03892925 0.02323187]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.8348240726520307
# GradientBoostingRegressor() 의 r2_score : 0.8348240726520307
# GradientBoostingRegressor() : [0.67058181 0.19487738 0.02133926 0.00984865 0.01235965 0.02768865
#  0.03011465 0.0234205  0.00976945]
# *************************************************
# XGBRegressor의 model.score: 0.8556410165986633
# XGBRegressor의 r2_score : 0.8556410165986633
# XGBRegressor: [0.3816536  0.11724094 0.31015724 0.02521955 0.03466619 0.03161917
#  0.03889856 0.03742571 0.02311898]

# [3,8]컬럼 제거 후 / 전반적으로 비슷함
# DecisionTreeRegressor() 의 model.score: 0.8989750755957694
# DecisionTreeRegressor() 의 r2_score : 0.8989750755957694
# DecisionTreeRegressor() : [0.5913537  0.20412717 0.03048663 0.06053294 0.04454634 0.03817931
#  0.03077389]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.9151176382339332
# RandomForestRegressor() 의 r2_score : 0.9151176382339332
# RandomForestRegressor() : [0.59345926 0.18710914 0.03488664 0.05818046 0.05459249 0.04356317
#  0.02820885]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.8175417799554847
# GradientBoostingRegressor() 의 r2_score : 0.8175417799554847
# GradientBoostingRegressor() : [0.6680074  0.20029284 0.01649595 0.0471115  0.03428521 0.0229741
#  0.01083299]
# *************************************************
# XGBRegressor의 model.score: 0.8373167735321537
# XGBRegressor의 r2_score : 0.8373167735321537
# XGBRegressor: [0.5677326  0.15611412 0.03407755 0.06256627 0.07392822 0.05533864
#  0.05024263]