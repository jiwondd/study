#https://www.kaggle.com/competitions/bike-sharing-demand/submit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

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

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)


# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)

#2. 모델구성
allAlgorithms=all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        scores=cross_val_score(model,x,y,cv=kfold)
    
        print(name,'의 R2:',scores)
        print('cross_val_score:', round(np.mean(scores),4))
    except:
        # continue
        print(name,'은 안나온 놈!!!')
        
# ARDRegression 의 R2: [0.33589042 0.35710881 0.34325987 0.34549936 0.34993956]
# cross_val_score: 0.3463
# AdaBoostRegressor 의 R2: [0.55467482 0.5755433  0.58266904 0.57898317 0.62101406]
# cross_val_score: 0.5826
# BaggingRegressor 의 R2: [0.83838463 0.84154537 0.84575865 0.84044272 0.83631065]
# cross_val_score: 0.8405
# BayesianRidge 의 R2: [0.33575593 0.35696643 0.34325795 0.34521349 0.35078307]
# cross_val_score: 0.3464
# CCA 의 R2: [0.12829561 0.21167513 0.13518296 0.13744604 0.16333514]
# cross_val_score: 0.1552
# DecisionTreeRegressor 의 R2: [0.70028228 0.72559355 0.73749293 0.74789097 0.71559709]
# cross_val_score: 0.7254
# DummyRegressor 의 R2: [-6.49419743e-04 -4.44297774e-04 -2.40830679e-05 -1.65108679e-03
#  -5.93256979e-08]
# cross_val_score: -0.0006
# ElasticNet 의 R2: [0.25439527 0.26211497 0.25568268 0.25045888 0.25436659]
# cross_val_score: 0.2554
# ElasticNetCV 의 R2: [0.33371215 0.35380936 0.33876431 0.33474781 0.34381841]
# cross_val_score: 0.341
# ExtraTreeRegressor 의 R2: [0.70206771 0.72985186 0.69478725 0.6929362  0.71020144]
# cross_val_score: 0.706
# ExtraTreesRegressor 의 R2: [0.8546692  0.85884661 0.85945675 0.8575004  0.84992158]
# cross_val_score: 0.8561
# GammaRegressor 의 R2: [0.14060181 0.1403444  0.135731   0.13975067 0.14135395]
# cross_val_score: 0.1396
# GaussianProcessRegressor 의 R2: [-1306.20326336  -301.98736464 -1497.12695088  -707.49861871
#   -295.31383526]
# cross_val_score: -821.626
# GradientBoostingRegressor 의 R2: [0.79316876 0.78366534 0.78986752 0.79913562 0.77085176]
# cross_val_score: 0.7873
# HistGradientBoostingRegressor 의 R2: [0.86760189 0.8684938  0.86360158 0.86917102 0.85307643]
# cross_val_score: 0.8644
# HuberRegressor 의 R2: [0.31465229 0.32549674 0.31382624 0.3093032  0.31886606]
# cross_val_score: 0.3164
# IsotonicRegression 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# KNeighborsRegressor 의 R2: [0.60561829 0.61765713 0.6305636  0.61438949 0.6232997 ]
# cross_val_score: 0.6183
# KernelRidge 의 R2: [0.33560724 0.3568192  0.34327868 0.34532246 0.35092403]
# cross_val_score: 0.3464
# Lars 의 R2: [0.32932815 0.19993656 0.33568902 0.34551792 0.35097638]
# cross_val_score: 0.3123
# LarsCV 의 R2: [0.33197203 0.349553   0.32688415 0.32477344 0.33708615]
# cross_val_score: 0.3341
# Lasso 의 R2: [0.33527623 0.35721904 0.34270984 0.34361387 0.34779896]
# cross_val_score: 0.3453
# LassoCV 의 R2: [0.33596981 0.35783127 0.34317146 0.34474872 0.34932187]
# cross_val_score: 0.3462
# LassoLars 의 R2: [-6.49419743e-04 -4.44297774e-04 -2.40830679e-05 -1.65108679e-03
#  -5.93256979e-08]
# cross_val_score: -0.0006
# LassoLarsCV 의 R2: [0.33608266 0.35744386 0.34329156 0.34496249 0.34897951]
# cross_val_score: 0.3462
# LassoLarsIC 의 R2: [0.33599131 0.35707421 0.34321888 0.34509433 0.35011273]
# cross_val_score: 0.3463
# LinearRegression 의 R2: [-5.85019233e+18  3.56761763e-01  3.43306860e-01  3.45399185e-01
#   3.50933148e-01]
# cross_val_score: -1.1700384659769925e+18
# LinearSVR 의 R2: [0.29095988 0.29902255 0.28554655 0.27808942 0.29119505]
# cross_val_score: 0.289
# MLPRegressor 의 R2: [0.52323433 0.54158813 0.55289355 0.54296246 0.54287386]
# cross_val_score: 0.5407
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskElasticNetCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLasso 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLassoCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# NuSVR 의 R2: [0.32913215 0.33544107 0.32930283 0.32039602 0.32658933]
# cross_val_score: 0.3282
# OrthogonalMatchingPursuit 의 R2: [0.15590607 0.16800949 0.15578748 0.16026013 0.16007406]      
# cross_val_score: 0.16
# OrthogonalMatchingPursuitCV 의 R2: [0.30661126 0.35272869 0.31691651 0.31263067 0.31937933]
# cross_val_score: 0.3217
# PLSCanonical 의 R2: [-0.45405245 -0.30716633 -0.37761751 -0.37515904 -0.33503572]
# cross_val_score: -0.3698
# PLSRegression 의 R2: [0.32528617 0.35207076 0.32741771 0.3252039  0.33680289]
# cross_val_score: 0.3334
# PassiveAggressiveRegressor 의 R2: [0.32114044 0.30041216 0.30117822 0.29177861 0.32515826]
# cross_val_score: 0.3079
# PoissonRegressor 의 R2: [0.36639696 0.38958882 0.36678296 0.37081994 0.3842011 ]
# cross_val_score: 0.3756
# RANSACRegressor 의 R2: [-7.11976167e+22  1.90966492e-01  1.40532969e-01  7.25165301e-02
#   1.98364885e-01]
# cross_val_score: -1.423952334953349e+22
# RadiusNeighborsRegressor 의 R2: [-8.65020278e+30 -5.90301642e+30 -4.77050669e+30 -7.00399571e+30
#  -4.73004074e+30]
# cross_val_score: -6.211552469930464e+30
# RandomForestRegressor 의 R2: [0.85453995 0.85662574 0.86140154 0.86049148 0.84773739]
# cross_val_score: 0.8562
# RegressorChain 은 안나온 놈!!!
# Ridge 의 R2: [0.33556813 0.356789   0.34330272 0.34537959 0.35091295]
# cross_val_score: 0.3464
# RidgeCV 의 R2: [0.33579807 0.35700591 0.34324859 0.34518787 0.35076257]
# cross_val_score: 0.3464
# SGDRegressor 의 R2: [0.3355712  0.35776049 0.3421213  0.34400551 0.34952639]
# cross_val_score: 0.3458
# SVR 의 R2: [0.31749281 0.3264535  0.31503169 0.29950533 0.31503139]
# cross_val_score: 0.3147
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 R2: [0.33502506 0.35657881 0.34092116 0.34274819 0.34901965]
# cross_val_score: 0.3449
# TransformedTargetRegressor 의 R2: [-5.85019233e+18  3.56761763e-01  3.43306860e-01  3.45399185e-01
#   3.50933148e-01]
# cross_val_score: -1.1700384659769925e+18
# TweedieRegressor 의 R2: [0.19570863 0.19991706 0.1960574  0.19128836 0.1941526 ]
# cross_val_score: 0.1954
# VotingRegressor 은 안나온 놈!!!