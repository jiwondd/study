#https://www.kaggle.com/competitions/bike-sharing-demand/submit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

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

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=777)
# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorithms=all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        model.fit(x_train,y_train)
    
        y_predict=model.predict(x_test)
        r2=r2_score(y_test,y_predict)
        print(name,'의 정답률 : ',r2)
    except:
        # continue
        print(name,'은 안나온 놈!!!')
        
# ARDRegression 의 정답률 :  0.21313066416872373
# AdaBoostRegressor 의 정답률 :  0.42544863796968824
# BaggingRegressor 의 정답률 :  0.8514961810999694
# BayesianRidge 의 정답률 :  0.21446481962260433
# CCA 의 정답률 :  -0.1290676210623758
# DecisionTreeRegressor 의 정답률 :  0.7637618466282434
# DummyRegressor 의 정답률 :  -0.01987107260561305
# ElasticNet 의 정답률 :  0.203513826216824
# ElasticNetCV 의 정답률 :  0.23069561304832698
# ExtraTreeRegressor 의 정답률 :  0.7041253083525474
# ExtraTreesRegressor 의 정답률 :  0.8518448801348842
# GammaRegressor 의 정답률 :  0.16897017458254027
# GaussianProcessRegressor 의 정답률 :  -9083.50186971285
# GradientBoostingRegressor 의 정답률 :  0.7701710523437548
# HistGradientBoostingRegressor 의 정답률 :  0.8296070735874223
# HuberRegressor 의 정답률 :  0.2213407005816571
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.45805667815483686
# KernelRidge 의 정답률 :  0.21419298483093963
# Lars 의 정답률 :  0.22081366604279484
# LarsCV 의 정답률 :  0.22320256855506027
# Lasso 의 정답률 :  0.22222515821943234
# LassoCV 의 정답률 :  0.21382828767268536
# LassoLars 의 정답률 :  -0.01987107260561305
# LassoLarsCV 의 정답률 :  0.21428713933730148
# LassoLarsIC 의 정답률 :  0.2160477386265982
# LinearRegression 의 정답률 :  0.21396065866589653
# LinearSVR 의 정답률 :  0.21097156625105096
# MLPRegressor 의 정답률 :  0.46665187920769446
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.29660511583020244
# OrthogonalMatchingPursuit 의 정답률 :  0.056705303164981014
# OrthogonalMatchingPursuitCV 의 정답률 :  0.20098491255659445
# PLSCanonical 의 정답률 :  -0.9177208946053623
# PLSRegression 의 정답률 :  0.21907171279242066
# PassiveAggressiveRegressor 의 정답률 :  0.22941660112013362
# PoissonRegressor 의 정답률 :  0.19633670657040136
# RANSACRegressor 의 정답률 :  0.038978186698084394
# RadiusNeighborsRegressor 의 정답률 :  -3.5307121530619565e+31
# RandomForestRegressor 의 정답률 :  0.8310037746253129
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.2140240556167906
# RidgeCV 의 정답률 :  0.2145544321994317
# SGDRegressor 의 정답률 :  0.21248521678064614
# SVR 의 정답률 :  0.27003788128724204
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.21080556765256875
# TransformedTargetRegressor 의 정답률 :  0.21396065866589653
# TweedieRegressor 의 정답률 :  0.15778034247886374
# VotingRegressor 은 안나온 놈!!!