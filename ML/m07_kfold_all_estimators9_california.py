from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
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


# ARDRegression 의 R2: [0.61652386 0.59257669 0.59486231 0.59820781 0.60728594]
# cross_val_score: 0.6019
# AdaBoostRegressor 의 R2: [0.36599277 0.4004188  0.44823101 0.48131522 0.4284403 ]
# cross_val_score: 0.4249
# BaggingRegressor 의 R2: [0.79551077 0.79204509 0.77604593 0.77947327 0.7878669 ]
# cross_val_score: 0.7862
# BayesianRidge 의 R2: [0.61660721 0.59291277 0.59476095 0.5980444  0.60723558]
# cross_val_score: 0.6019
# CCA 의 R2: [0.5787713  0.55665247 0.55316522 0.55422957 0.56913271]
# cross_val_score: 0.5624
# DecisionTreeRegressor 의 R2: [0.62376348 0.60567129 0.61012093 0.60572225 0.61800268]        
# cross_val_score: 0.6127
# DummyRegressor 의 R2: [-3.24186381e-04 -5.02433932e-05 -7.66495474e-06 -8.44220662e-05       
#  -1.44274084e-03]
# cross_val_score: -0.0004
# ElasticNet 의 R2: [-3.24186381e-04 -5.02433932e-05 -7.66495474e-06 -8.44220662e-05
#  -1.44274084e-03]
# cross_val_score: -0.0004
# ElasticNetCV 의 R2: [0.61721602 0.60137345 0.58438498 0.58621143 0.5987305 ]
# cross_val_score: 0.5976
# ExtraTreeRegressor 의 R2: [0.55716331 0.54056538 0.58351644 0.65085916 0.57153152]
# cross_val_score: 0.5807
# ExtraTreesRegressor 의 R2: [0.82399872 0.81474437 0.80174796 0.80788216 0.81344722]        
# cross_val_score: 0.8124
# GammaRegressor 의 R2: [0.01974024 0.01930877 0.01886774 0.01843498 0.01805743]
# cross_val_score: 0.0189
# GaussianProcessRegressor 의 R2: [-1.92762653e+04 -1.33209195e+02 -6.61839682e+01  6.53192745e-01
#  -2.20576331e+02]
# cross_val_score: -3939.1163
# GradientBoostingRegressor 의 R2: [0.80092672 0.78567305 0.77628392 0.78151714 0.79112813]
# cross_val_score: 0.7871
# HistGradientBoostingRegressor 의 R2: [0.84664809 0.83549698 0.82720845 0.82873858 0.83991468]
# cross_val_score: 0.8356
# HuberRegressor 의 R2: [0.57757621 0.57357787 0.57788964 0.59206293 0.60328166]
# cross_val_score: 0.5849
# IsotonicRegression 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# KNeighborsRegressor 의 R2: [0.7072116  0.70569964 0.68756324 0.69045312 0.7106806 ]
# cross_val_score: 0.7003
# KernelRidge 의 R2: [0.54862516 0.5296672  0.51322098 0.52019046 0.53464008]
# cross_val_score: 0.5293
# Lars 의 R2: [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957]
# cross_val_score: 0.6018
# LarsCV 의 R2: [0.61666036 0.59250746 0.42823193 0.5981755  0.60682002]
# cross_val_score: 0.5685
# Lasso 의 R2: [-3.24186381e-04 -5.02433932e-05 -7.66495474e-06 -8.44220662e-05
#  -1.44274084e-03]
# cross_val_score: -0.0004
# LassoCV 의 R2: [0.6234529  0.59851804 0.5898343  0.5950542  0.60598601]
# cross_val_score: 0.6026
# LassoLars 의 R2: [-3.24186381e-04 -5.02433932e-05 -7.66495474e-06 -8.44220662e-05
#  -1.44274084e-03]
# cross_val_score: -0.0004
# LassoLarsCV 의 R2: [0.61666036 0.59250746 0.42823193 0.5981755  0.60682002]
# cross_val_score: 0.5685
# LassoLarsIC 의 R2: [0.61614066 0.59347925 0.59471286 0.5981755  0.60724957]
# cross_val_score: 0.602
# LinearRegression 의 R2: [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957]
# cross_val_score: 0.6018
# LinearSVR 의 R2: [0.60112933 0.58353968 0.56923169 0.56790929 0.5885318 ]
# cross_val_score: 0.5821
# MLPRegressor 의 R2: [0.7157627  0.70392791 0.68708324 0.71535284 0.69145815]
# cross_val_score: 0.7027
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskElasticNetCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLasso 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLassoCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# NuSVR 의 R2: [0.68393373 0.662099   0.64559495 0.64736907 0.67080767]
# cross_val_score: 0.662
# OrthogonalMatchingPursuit 의 R2: [0.49497108 0.47987668 0.45729284 0.45909253 0.47373082]
# cross_val_score: 0.473
# OrthogonalMatchingPursuitCV 의 R2: [0.61783292 0.59933302 0.49332567 0.58212067 0.59210345]cross_val_score: 0.5769
# PLSCanonical 의 R2: [0.35849242 0.37071128 0.37797438 0.38564743 0.36020325]
# cross_val_score: 0.3706
# PLSRegression 의 R2: [0.52503134 0.52940822 0.51350893 0.51448473 0.52158067]
# cross_val_score: 0.5208
# PassiveAggressiveRegressor 의 R2: [ 0.46013692  0.58741451  0.28215382 -0.17056586  0.2519242 ]
# cross_val_score: 0.2822
# PoissonRegressor 의 R2: [0.04232559 0.04137103 0.0401655  0.03979503 0.03997003]
# cross_val_score: 0.0407
# RANSACRegressor 의 R2: [ 0.20197461  0.34412273 -0.10931199  0.39921051  0.3114544 ]
# cross_val_score: 0.2295
# RadiusNeighborsRegressor 의 R2: [0.01456194 0.01417624 0.01298869 0.01311859 0.01292349]
# cross_val_score: 0.0136
# RandomForestRegressor 의 R2: [0.82374603 0.81463285 0.79579149 0.80537224 0.80825119]
# cross_val_score: 0.8096
# RegressorChain 은 안나온 놈!!!
# Ridge 의 R2: [0.62009751 0.60213035 0.58677334 0.58887479 0.60101638]
# cross_val_score: 0.5998
# RidgeCV 의 R2: [0.62131905 0.59876463 0.593135   0.59612889 0.60646719]
# cross_val_score: 0.6032
# SGDRegressor 의 R2: [0.58359567 0.56933437 0.54777993 0.55159095 0.56822999]
# cross_val_score: 0.5641
# SVR 의 R2: [0.6791983  0.65717416 0.64094897 0.64227206 0.66726156]
# cross_val_score: 0.6574
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 R2: [-38.80661708   0.25467712  -8.23685883   0.61497487   0.50952367]cross_val_score: -9.1329        
# TransformedTargetRegressor 의 R2: [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957]
# cross_val_score: 0.6018
# TweedieRegressor 의 R2: [0.01951281 0.01935974 0.01880883 0.01870162 0.01822075]
# cross_val_score: 0.0189
# VotingRegressor 은 안나온 놈!!!
