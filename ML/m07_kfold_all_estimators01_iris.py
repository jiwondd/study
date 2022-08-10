import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings

warnings.filterwarnings('ignore')

#1. 데이터 
datasets=load_iris()
x=datasets['data']
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

allAlgorithms=all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        scores=cross_val_score(model,x,y,cv=kfold)
    
        print(name,'의 acc_score:',scores)
        print('cross_val_score:', round(np.mean(scores),4))
    except:
        # continue
        print(name,'은 안나온 놈!!!')
    
# AdaBoostClassifier 의 acc_score: [0.63333333 0.93333333 1.         0.9        0.96666667]
# cross_val_score: 0.8867
# BaggingClassifier 의 acc_score: [0.96666667 0.96666667 1.         0.86666667 0.96666667]
# cross_val_score: 0.9533
# BernoulliNB 의 acc_score: [0.36666667 0.36666667 0.3        0.36666667 0.4       ]
# cross_val_score: 0.36
# CalibratedClassifierCV 의 acc_score: [0.96666667 0.8        1.         0.83333333 0.93333333]
# cross_val_score: 0.9067
# CategoricalNB 의 acc_score: [0.33333333        nan        nan 0.23333333 0.3       ]
# cross_val_score: nan
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 acc_score: [0.66666667 0.66666667 0.7        0.6        0.7       ]
# cross_val_score: 0.6667
# DecisionTreeClassifier 의 acc_score: [0.93333333 0.96666667 1.         0.9        0.93333333]  
# cross_val_score: 0.9467
# DummyClassifier 의 acc_score: [0.3        0.33333333 0.3        0.23333333 0.3       ]
# cross_val_score: 0.2933
# ExtraTreeClassifier 의 acc_score: [0.96666667 0.96666667 1.         0.86666667 0.96666667]     
# cross_val_score: 0.9533
# ExtraTreesClassifier 의 acc_score: [0.93333333 0.96666667 1.         0.86666667 0.96666667]
# cross_val_score: 0.9467
# GaussianNB 의 acc_score: [0.96666667 0.9        1.         0.9        0.96666667]
# cross_val_score: 0.9467
# GaussianProcessClassifier 의 acc_score: [0.96666667 0.86666667 1.         0.86666667 0.96666667]
# cross_val_score: 0.9333
# GradientBoostingClassifier 의 acc_score: [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# cross_val_score: 0.9667
# HistGradientBoostingClassifier 의 acc_score: [0.86666667 0.96666667 1.         0.9        0.96666667]      
# cross_val_score: 0.94
# KNeighborsClassifier 의 acc_score: [1.         0.96666667 1.         0.86666667 0.96666667]
# cross_val_score: 0.96
# LabelPropagation 의 acc_score: [0.96666667 0.96666667 1.         0.9        0.96666667]
# cross_val_score: 0.96
# LabelSpreading 의 acc_score: [0.96666667 0.96666667 1.         0.9        0.96666667]
# cross_val_score: 0.96
# LinearDiscriminantAnalysis 의 acc_score: [1.  1.  1.  0.9 1. ]
# cross_val_score: 0.98
# LinearSVC 의 acc_score: [0.96666667 0.83333333 1.         0.83333333 0.96666667]
# cross_val_score: 0.92
# LogisticRegression 의 acc_score: [0.96666667 0.86666667 1.         0.86666667 0.96666667]
# cross_val_score: 0.9333
# LogisticRegressionCV 의 acc_score: [1.         0.93333333 1.         0.9        0.96666667]
# cross_val_score: 0.96
# MLPClassifier 의 acc_score: [0.96666667 0.93333333 0.96666667 0.86666667 0.96666667]
# cross_val_score: 0.94
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 acc_score: [0.63333333 0.9        0.76666667 0.56666667 0.86666667]
# cross_val_score: 0.7467
# NearestCentroid 의 acc_score: [0.96666667 0.9        0.96666667 0.9        0.96666667]
# cross_val_score: 0.94
# NuSVC 의 acc_score: [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# cross_val_score: 0.9667
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 acc_score: [0.9        0.86666667 1.         0.83333333 0.96666667]
# cross_val_score: 0.9133
# Perceptron 의 acc_score: [0.9        0.8        1.         0.86666667 0.8       ]
# cross_val_score: 0.8733
# QuadraticDiscriminantAnalysis 의 acc_score: [1.         0.96666667 1.         0.93333333 1.        ]       
# cross_val_score: 0.98
# RadiusNeighborsClassifier 의 acc_score: [0.46666667 0.56666667 0.4        0.56666667 0.36666667]
# cross_val_score: 0.4733
# RandomForestClassifier 의 acc_score: [0.96666667 0.96666667 1.         0.9        0.96666667]
# cross_val_score: 0.96
# RidgeClassifier 의 acc_score: [0.93333333 0.8        0.93333333 0.73333333 0.9       ]
# cross_val_score: 0.86
# RidgeClassifierCV 의 acc_score: [0.83333333 0.8        0.93333333 0.73333333 0.93333333]
# cross_val_score: 0.8467
# SGDClassifier 의 acc_score: [0.93333333 0.9        1.         0.93333333 0.93333333]
# cross_val_score: 0.94
# SVC 의 acc_score: [1.         0.96666667 1.         0.86666667 0.96666667]
# cross_val_score: 0.96
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!

    
