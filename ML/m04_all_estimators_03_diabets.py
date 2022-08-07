from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=72)

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
        
# ARDRegression 의 정답률 :  0.6262017478844342
# AdaBoostRegressor 의 정답률 :  0.5454453330961967
# BaggingRegressor 의 정답률 :  0.5126542578405786
# BayesianRidge 의 정답률 :  0.6226295803821278
# CCA 의 정답률 :  0.6353179985243622
# DecisionTreeRegressor 의 정답률 :  -0.022224903421618647
# DummyRegressor 의 정답률 :  -0.0011144203392519092
# ElasticNet 의 정답률 :  0.50800622188649
# ElasticNetCV 의 정답률 :  0.6213285434019378
# ExtraTreeRegressor 의 정답률 :  -0.09445136668225729
# ExtraTreesRegressor 의 정답률 :  0.568045975384136
# GammaRegressor 의 정답률 :  0.43532290311597777
# GaussianProcessRegressor 의 정답률 :  0.035503808034670015
# GradientBoostingRegressor 의 정답률 :  0.5616258520809374
# HistGradientBoostingRegressor 의 정답률 :  0.5449287423096122
# HuberRegressor 의 정답률 :  0.6296874205003458
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.5477564896710819
# KernelRidge 의 정답률 :  -1.0400441102861162
# Lars 의 정답률 :  0.6229230186189666
# LarsCV 의 정답률 :  0.6268464473041175
# Lasso 의 정답률 :  0.6214715937648331
# LassoCV 의 정답률 :  0.6310550510704772
# LassoLars 의 정답률 :  0.43602959555310616
# LassoLarsCV 의 정답률 :  0.6310634453636433
# LassoLarsIC 의 정답률 :  0.631011387486286
# LinearRegression 의 정답률 :  0.6307176518001985
# LinearSVR 의 정답률 :  0.302434114651837
# MLPRegressor 의 정답률 :  -1.0586429097190622
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.16148498534815914
# OrthogonalMatchingPursuit 의 정답률 :  0.43235114462770363
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6292810126044563
# PLSCanonical 의 정답률 :  -1.1687994059745086
# PLSRegression 의 정답률 :  0.6189994279765478
# PassiveAggressiveRegressor 의 정답률 :  0.5587693367114437
# PoissonRegressor 의 정답률 :  0.6342002574183969
# RANSACRegressor 의 정답률 :  0.3014137956876404
# RadiusNeighborsRegressor 은 안나온 놈!!!
# RandomForestRegressor 의 정답률 :  0.5546091580130517
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.6305059873099594
# RidgeCV 의 정답률 :  0.6223046512499371
# SGDRegressor 의 정답률 :  0.6279467898892199
# SVR 의 정답률 :  0.1553399396619206
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.6241236976832021
# TransformedTargetRegressor 의 정답률 :  0.6307176518001985
# TweedieRegressor 의 정답률 :  0.4405462829370098
# VotingRegressor 은 안나온 놈!!!
        