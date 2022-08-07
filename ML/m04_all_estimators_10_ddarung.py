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
x=train_set.drop(['count'],axis=1)
y=train_set['count']

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=750)

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
        
# ARDRegression 의 정답률 :  0.8028840548811713
# AdaBoostRegressor 의 정답률 :  0.5139982533595211
# BaggingRegressor 의 정답률 :  0.9063361873559941
# BayesianRidge 의 정답률 :  0.7953757733715093
# CCA 의 정답률 :  0.4517084377785968
# DecisionTreeRegressor 의 정답률 :  0.8897984268602167
# DummyRegressor 의 정답률 :  -0.035829707903143326
# ElasticNet 의 정답률 :  0.5791059516831218
# ElasticNetCV 의 정답률 :  0.7668272838997392
# ExtraTreeRegressor 의 정답률 :  0.8306164034640856
# ExtraTreesRegressor 의 정답률 :  0.9121543216944669
# GammaRegressor 의 정답률 :  0.4623021257948525
# GaussianProcessRegressor 의 정답률 :  -0.1668243993641727
# GradientBoostingRegressor 의 정답률 :  0.8348240726520307
# HistGradientBoostingRegressor 의 정답률 :  0.8919489551363928
# HuberRegressor 의 정답률 :  0.8154456364478878
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.8650296541055397
# KernelRidge 의 정답률 :  -0.8968750833765333
# Lars 의 정답률 :  0.7981674407799787
# LarsCV 의 정답률 :  0.7995417885702589
# Lasso 의 정답률 :  0.7929057348204154
# LassoCV 의 정답률 :  0.7977873726971476
# LassoLars 의 정답률 :  0.34993613951764
# LassoLarsCV 의 정답률 :  0.7995887603548828
# LassoLarsIC 의 정답률 :  0.7994778967627575
# LinearRegression 의 정답률 :  0.7981674407799786
# LinearSVR 의 정답률 :  0.754884696482131
# MLPRegressor 의 정답률 :  0.7590281116226517
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.541262800669261
# OrthogonalMatchingPursuit 의 정답률 :  0.8345259391253743
# OrthogonalMatchingPursuitCV 의 정답률 :  0.7905921501860076
# PLSCanonical 의 정답률 :  -0.7425667016585928
# PLSRegression 의 정답률 :  0.8154697258734681
# PassiveAggressiveRegressor 의 정답률 :  0.6471945106740882
# PoissonRegressor 의 정답률 :  0.8139962363698907
# RANSACRegressor 의 정답률 :  0.7796647679504403
# RadiusNeighborsRegressor 의 정답률 :  0.8493849963730838
# RandomForestRegressor 의 정답률 :  0.915635901330579
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.7973082524610132
# RidgeCV 의 정답률 :  0.7973082524605595
# SGDRegressor 의 정답률 :  0.796756853491098
# SVR 의 정답률 :  0.5802956599592075
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.825587969633934
# TransformedTargetRegressor 의 정답률 :  0.7981674407799786
# TweedieRegressor 의 정답률 :  0.47780432041789234
# VotingRegressor 은 안나온 놈!!!