import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from icecream import ic
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count','hour_bef_precipitation','hour_bef_humidity'],axis=1)
y=train_set['count']

# print(x.shape) #(1328, 9)->(1328, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=1234)

kfold=KFold(n_splits=5,shuffle=True,random_state=123)

# 2.모델
model=make_pipeline(StandardScaler(),
                    RandomForestRegressor()
                    )

model.fit(x_train,y_train)

score=model.score(x_test,y_test)
cv_score=cross_val_score(model,x_train, y_train, cv=kfold,scoring='r2')
ic(score) #score: 0.7665382927362877 (poly적용전)
ic(cv_score,np.mean(cv_score))

############polynomial적용후###############
pf=PolynomialFeatures(degree=2) #include_bias=False
xp=pf.fit_transform(x)


x_train2,x_test2,y_train2,y_test2=train_test_split(xp,y,
        train_size=0.8,shuffle=True, random_state=1234)

# 2.모델
model2=make_pipeline(StandardScaler(),
                    RandomForestRegressor()
                    )

model.fit(x_train2,y_train2)

poly_score=model.score(x_test2,y_test2)
cv_score2=cross_val_score(model2,x_train2, y_train2, cv=kfold,scoring='r2')
ic(poly_score) 
ic(cv_score2,np.mean(cv_score2))

# ic| score: 0.7570984989295129
# ic| cv_score: array([0.76191314, 0.75388189, 0.75129877, 0.73253875, 0.75815983])
#     np.mean(cv_score): 0.7515584752026043

# ic| poly_score: 0.7820958261930001
# ic| cv_score2: array([0.78718814, 0.70833697, 0.68265326, 0.71603767, 0.69215927])
#     np.mean(cv_score2): 0.7172750604317495
