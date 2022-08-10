import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
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
        
# ARDRegression 의 R2: [0.51077965 0.5915414  0.60643283 0.58823325 0.60363204]
# cross_val_score: 0.5801
# AdaBoostRegressor 의 R2: [0.49058663 0.57375535 0.67428134 0.62158734 0.55768005]
# cross_val_score: 0.5836
# BaggingRegressor 의 R2: [0.73342477 0.69691073 0.77853929 0.74798364 0.75368485]
# cross_val_score: 0.7421
# BayesianRidge 의 R2: [0.50693999 0.58859071 0.60877452 0.59078513 0.6052332 ]
# cross_val_score: 0.5801
# CCA 의 R2: [-0.05472715  0.12092942  0.38457029  0.29610656  0.17217105]     
# cross_val_score: 0.1838
# DecisionTreeRegressor 의 R2: [0.47121711 0.52044805 0.69171129 0.49587136 0.51997121] 
# cross_val_score: 0.5398
# DummyRegressor 의 R2: [-1.06844486e-06 -8.22274615e-03 -3.89996785e-03 -8.86961127e-03
#  -5.56153108e-03]
# cross_val_score: -0.0053
# ElasticNet 의 R2: [0.44419548 0.49546762 0.5009876  0.50125898 0.51201212]
# cross_val_score: 0.4908
# ElasticNetCV 의 R2: [0.50211007 0.58053024 0.60126183 0.58840913 0.60097541]
# cross_val_score: 0.5747
# ExtraTreeRegressor 의 R2: [0.49985403 0.55363897 0.6768851  0.50167657 0.51311286]
# cross_val_score: 0.549
# ExtraTreesRegressor 의 R2: [0.78753309 0.78932252 0.83046323 0.79326611 0.80691746]
# cross_val_score: 0.8015
# GammaRegressor 의 R2: [0.29953267 0.31274251 0.29470774 0.32907088 0.33686923]
# cross_val_score: 0.3146
# GaussianProcessRegressor 의 R2: [-0.98953932 -0.09517257 -0.19011329 -0.41409733 -0.07224358]
# cross_val_score: -0.3522
# GradientBoostingRegressor 의 R2: [0.74259808 0.72959076 0.79271488 0.77966686 0.78021526]
# cross_val_score: 0.765
# HistGradientBoostingRegressor 의 R2: [0.74823854 0.7725782  0.8287384  0.79532989 0.81611725]
# cross_val_score: 0.7922
# HuberRegressor 의 R2: [0.5049127  0.58262308 0.59272644 0.57042134 0.60855019]
# cross_val_score: 0.5718
# IsotonicRegression 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# KNeighborsRegressor 의 R2: [0.63475627 0.65392425 0.70778499 0.58817946 0.70752893]
# cross_val_score: 0.6584
# KernelRidge 의 R2: [-1.38141528 -1.27690989 -0.85291756 -1.09570846 -0.93031792]
# cross_val_score: -1.1075
# Lars 의 R2: [0.50717332 0.58924277 0.60930529 0.59047738 0.60451426]
# cross_val_score: 0.5801
# LarsCV 의 R2: [0.50717332 0.58924277 0.59095966 0.58875565 0.60472822]
# cross_val_score: 0.5762
# Lasso 의 R2: [0.51128815 0.58079743 0.59622705 0.5827469  0.60247832]
# cross_val_score: 0.5747
# LassoCV 의 R2: [0.50755344 0.58930586 0.60849667 0.59023826 0.60461756]
# cross_val_score: 0.58
# LassoLars 의 R2: [0.33620042 0.32393138 0.28736021 0.29995657 0.33061674]
# cross_val_score: 0.3156
# LassoLarsCV 의 R2: [0.50717332 0.58924277 0.6031874  0.58875565 0.60472822]
# cross_val_score: 0.5786
# LassoLarsIC 의 R2: [0.51273251 0.5914517  0.60726859 0.58888997 0.60417292]
# cross_val_score: 0.5809
# LinearRegression 의 R2: [0.50717332 0.58924277 0.60930529 0.59047738 0.60451426]
# cross_val_score: 0.5801
# LinearSVR 의 R2: [0.45698755 0.53207869 0.52656398 0.50616308 0.56570949]
# cross_val_score: 0.5175
# MLPRegressor 의 R2: [0.49743334 0.53984206 0.6489177  0.5737414  0.59625332]
# cross_val_score: 0.5712
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskElasticNetCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLasso 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLassoCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# NuSVR 의 R2: [0.3696572  0.42404998 0.42512011 0.40305056 0.4689005 ]
# cross_val_score: 0.4182
# OrthogonalMatchingPursuit 의 R2: [0.28281675 0.31546766 0.369908   0.33529211 0.39700932]      
# cross_val_score: 0.3401
# OrthogonalMatchingPursuitCV 의 R2: [0.49789496 0.579448   0.59113518 0.57432654 0.59866295]
# cross_val_score: 0.5683
# PLSCanonical 의 R2: [-0.92642071 -0.55170831 -0.09798047 -0.1505343  -0.60939897]
# cross_val_score: -0.4672
# PLSRegression 의 R2: [0.50037152 0.56957989 0.60371941 0.58488749 0.60480898]
# cross_val_score: 0.5727
# PassiveAggressiveRegressor 의 R2: [0.48070614 0.57657645 0.57321588 0.55392246 0.56536689]     
# cross_val_score: 0.55
# PoissonRegressor 의 R2: [0.50900314 0.5978671  0.62392632 0.60934977 0.63814144]
# cross_val_score: 0.5957
# RANSACRegressor 의 R2: [0.41513188 0.47236431 0.56949281 0.3949011  0.56922003]
# cross_val_score: 0.4842
# RadiusNeighborsRegressor 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# RandomForestRegressor 의 R2: [0.75699856 0.72429639 0.81659201 0.7790181  0.79761271]
# cross_val_score: 0.7749
# RegressorChain 은 안나온 놈!!!
# Ridge 의 R2: [0.5071215  0.58907799 0.60916216 0.59060406 0.60480701]
# cross_val_score: 0.5802
# RidgeCV 의 R2: [0.5071215  0.58907799 0.60916216 0.59060406 0.60480701]
# cross_val_score: 0.5802
# SGDRegressor 의 R2: [0.5068102  0.58746891 0.60888418 0.58954968 0.60618991]
# cross_val_score: 0.5798
# SVR 의 R2: [0.36432392 0.41875739 0.42599511 0.40541292 0.47729789]
# cross_val_score: 0.4184
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 R2: [0.50859318 0.56937479 0.58684053 0.56947382 0.59590898]
# cross_val_score: 0.566
# TransformedTargetRegressor 의 R2: [0.50717332 0.58924277 0.60930529 0.59047738 0.60451426]
# cross_val_score: 0.5801
# TweedieRegressor 의 R2: [0.39167152 0.42988045 0.42820842 0.42963516 0.44282103]
# cross_val_score: 0.4244
# VotingRegressor 은 안나온 놈!!!