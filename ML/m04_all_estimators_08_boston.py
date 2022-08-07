import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

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
        
# ARDRegression 의 정답률 :  0.7001069826832631
# AdaBoostRegressor 의 정답률 :  0.8112025853848824
# BaggingRegressor 의 정답률 :  0.8547214955688119
# BayesianRidge 의 정답률 :  0.7078894761884061
# CCA 의 정답률 :  0.6671691420898105
# DecisionTreeRegressor 의 정답률 :  0.7862815256995472
# DummyRegressor 의 정답률 :  -0.000855141696997741
# ElasticNet 의 정답률 :  0.6185797666038255
# ElasticNetCV 의 정답률 :  0.7131882188004979
# ExtraTreeRegressor 의 정답률 :  0.6123393834639235
# ExtraTreesRegressor 의 정답률 :  0.8706438654599731
# GammaRegressor 의 정답률 :  0.5911583521912034
# GaussianProcessRegressor 의 정답률 :  0.327863849849066
# GradientBoostingRegressor 의 정답률 :  0.8707728441946535
# HistGradientBoostingRegressor 의 정답률 :  0.8604833005976018
# HuberRegressor 의 정답률 :  0.7030639407261698
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.7209948731531117
# KernelRidge 의 정답률 :  -1.7854526671329154
# Lars 의 정답률 :  0.7000498170510021
# LarsCV 의 정답률 :  0.7057968079888159
# Lasso 의 정답률 :  0.6729492162385879
# LassoCV 의 정답률 :  0.7051628540139496
# LassoLars 의 정답률 :  -0.000855141696997741
# LassoLarsCV 의 정답률 :  0.7043360805578183
# LassoLarsIC 의 정답률 :  0.7012983234498744
# LinearRegression 의 정답률 :  0.7000498170510017
# LinearSVR 의 정답률 :  0.6889229036834066
# MLPRegressor 의 정답률 :  0.4715264107790592
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.5800019771198888
# OrthogonalMatchingPursuit 의 정답률 :  0.5132429351213895
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6810381191307746
# PLSCanonical 의 정답률 :  -2.256646444232582
# PLSRegression 의 정답률 :  0.7107306986172064
# PassiveAggressiveRegressor 의 정답률 :  0.36031027262958704
# PoissonRegressor 의 정답률 :  0.7774385181048213
# RANSACRegressor 의 정답률 :  -0.4451458222834994
# RadiusNeighborsRegressor 은 안나온 놈!!!
# RandomForestRegressor 의 정답률 :  0.8611100892482083
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.7043016610386998
# RidgeCV 의 정답률 :  0.7043016610386134
# SGDRegressor 의 정답률 :  0.7134476976102877
# SVR 의 정답률 :  0.6014040127688285
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.689715427440226
# TransformedTargetRegressor 의 정답률 :  0.7000498170510017
# TweedieRegressor 의 정답률 :  0.5879444143051753
# VotingRegressor 은 안나온 놈!!!