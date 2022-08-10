import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target
n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
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

        
# ARDRegression 의 R2: [0.81190155 0.78214562 0.5901016  0.64004994 0.71461803]
# cross_val_score: 0.7078
# AdaBoostRegressor 의 R2: [0.88159261 0.8000857  0.77178765 0.83141268 0.86931702]
# cross_val_score: 0.8308
# BaggingRegressor 의 R2: [0.91339375 0.8707546  0.79197536 0.897666   0.91012402]
# cross_val_score: 0.8768
# BayesianRidge 의 R2: [0.81243191 0.79917585 0.59370086 0.64027728 0.72256599]
# cross_val_score: 0.7136
# CCA 의 R2: [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276]
# cross_val_score: 0.6471
# DecisionTreeRegressor 의 R2: [0.78891578 0.6696454  0.76276567 0.6992975  0.71708843]
# cross_val_score: 0.7275
# DummyRegressor 의 R2: [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
# cross_val_score: -0.0135
# ElasticNet 의 R2: [0.16194042 0.14965471 0.16407777 0.11801086 0.13494007]
# cross_val_score: 0.1457
# ElasticNetCV 의 R2: [0.78568701 0.78487688 0.61006443 0.62768954 0.70632912]
# cross_val_score: 0.7029
# ExtraTreeRegressor 의 R2: [0.84067721 0.53467152 0.48664665 0.71893866 0.77518141]
# cross_val_score: 0.6712
# ExtraTreesRegressor 의 R2: [0.93278731 0.8563062  0.78043353 0.88004243 0.92162383]
# cross_val_score: 0.8742
# GammaRegressor 의 R2: [0.22076246 0.18445107 0.19021682 0.14387293 0.17909782]
# cross_val_score: 0.1837
# GaussianProcessRegressor 의 R2: [-1.46607898 -0.397381   -1.44972663 -1.67202053 -1.65318792]
# cross_val_score: -1.3277
# GradientBoostingRegressor 의 R2: [0.94514113 0.83906518 0.82634102 0.88652156 0.93211293]
# cross_val_score: 0.8858
# HistGradientBoostingRegressor 의 R2: [0.93233261 0.82405072 0.78738013 0.88879806 0.85753969]
# cross_val_score: 0.858
# HuberRegressor 의 R2: [0.79584249 0.7697038  0.5989051  0.58057011 0.72395616]
# cross_val_score: 0.6938
# IsotonicRegression 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# KNeighborsRegressor 의 R2: [0.83348102 0.87007238 0.59230238 0.5418837  0.73111046]
# cross_val_score: 0.7138
# KernelRidge 의 R2: [0.80308658 0.72505694 0.51081446 0.51457727 0.65642107]
# cross_val_score: 0.642
# Lars 의 R2: [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384]
# cross_val_score: 0.6977
# LarsCV 의 R2: [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854]
# cross_val_score: 0.6928
# Lasso 의 R2: [0.24248548 0.2224108  0.27604132 0.20682847 0.20626446]
# cross_val_score: 0.2308
# LassoCV 의 R2: [0.80540771 0.77789687 0.59356913 0.63336116 0.72054328]
# cross_val_score: 0.7062
# LassoLars 의 R2: [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
# cross_val_score: -0.0135
# LassoLarsCV 의 R2: [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787]
# cross_val_score: 0.6965
# LassoLarsIC 의 R2: [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009]
# cross_val_score: 0.713
# LinearRegression 의 R2: [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
# cross_val_score: 0.7128
# LinearSVR 의 R2: [0.70847914 0.69925223 0.59004641 0.47797344 0.62082802]
# cross_val_score: 0.6193
# MLPRegressor 의 R2: [0.44499153 0.39589371 0.25889925 0.23221307 0.40385205]
# cross_val_score: 0.3472
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskElasticNetCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLasso 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLassoCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# NuSVR 의 R2: [0.62706487 0.67823799 0.57019971 0.45564557 0.53395381]
# cross_val_score: 0.573
# OrthogonalMatchingPursuit 의 R2: [0.58276176 0.565867   0.48689774 0.51545117 0.52049576]
# cross_val_score: 0.5343
# OrthogonalMatchingPursuitCV 의 R2: [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377]
# cross_val_score: 0.6578
# PLSCanonical 의 R2: [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868]
# cross_val_score: -2.2096
# PLSRegression 의 R2: [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313]
# cross_val_score: 0.6847
# PassiveAggressiveRegressor 의 R2: [0.58474876 0.62134279 0.595216   0.58846084 0.72878237]
# cross_val_score: 0.6237
# PoissonRegressor 의 R2: [0.69963871 0.67198278 0.5904302  0.53884814 0.61357309]
# cross_val_score: 0.6229
# RANSACRegressor 의 R2: [0.51506238 0.09229482 0.5490162  0.46445104 0.48216264]
# cross_val_score: 0.4206
# RadiusNeighborsRegressor 의 R2: [0.41192592 0.39498389 0.26429904 0.20900803 0.41015248]
# cross_val_score: 0.3381
# RandomForestRegressor 의 R2: [0.92660611 0.85780411 0.81671348 0.87961241 0.90430914]
# cross_val_score: 0.877
# RegressorChain 은 안나온 놈!!!
# Ridge 의 R2: [0.80936111 0.79802632 0.60378155 0.6363263  0.71693218]
# cross_val_score: 0.7129
# RidgeCV 의 R2: [0.81189808 0.79882098 0.59220545 0.64058078 0.72303833]
# cross_val_score: 0.7133
# SGDRegressor 의 R2: [0.81887449 0.79078695 0.57318482 0.60733504 0.70806428]
# cross_val_score: 0.6996
# SVR 의 R2: [0.66164948 0.72226578 0.60102126 0.47371442 0.55646757]
# cross_val_score: 0.603
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 R2: [0.79035415 0.75547581 0.59807966 0.55750715 0.71415216]
# cross_val_score: 0.6831
# TransformedTargetRegressor 의 R2: [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
# cross_val_score: 0.7128
# TweedieRegressor 의 R2: [0.19438607 0.18409052 0.18186094 0.13487259 0.16130085]
# cross_val_score: 0.1713
# VotingRegressor 은 안나온 놈!!!