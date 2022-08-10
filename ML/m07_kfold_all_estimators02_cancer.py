import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target

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
        
# AdaBoostClassifier 의 acc_score: [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133]
# cross_val_score: 0.9649
# BaggingClassifier 의 acc_score: [0.93859649 0.95614035 0.93859649 0.93859649 0.97345133]
# cross_val_score: 0.9491
# BernoulliNB 의 acc_score: [0.61403509 0.64912281 0.61403509 0.59649123 0.61946903]
# cross_val_score: 0.6186
# CalibratedClassifierCV 의 acc_score: [0.97368421 1.         0.94736842 0.96491228 0.98230088]
# cross_val_score: 0.9737
# CategoricalNB 의 acc_score: [nan nan nan nan nan]
# cross_val_score: nan
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 acc_score: [0.79824561 0.89473684 0.90350877 0.85087719 0.83185841]
# cross_val_score: 0.8558
# DecisionTreeClassifier 의 acc_score: [0.93859649 0.92105263 0.92105263 0.88596491 0.92920354]
# cross_val_score: 0.9192
# DummyClassifier 의 acc_score: [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
# cross_val_score: 0.6274
# ExtraTreeClassifier 의 acc_score: [0.92982456 0.9122807  0.9122807  0.93859649 0.94690265]
# cross_val_score: 0.928
# ExtraTreesClassifier 의 acc_score: [0.96491228 0.97368421 0.96491228 0.95614035 0.97345133]
# cross_val_score: 0.9666
# GaussianNB 의 acc_score: [0.92105263 0.96491228 0.92982456 0.92105263 0.92920354]
# cross_val_score: 0.9332
# GaussianProcessClassifier 의 acc_score: [0.96491228 0.98245614 0.95614035 0.94736842 0.96460177]
# cross_val_score: 0.9631
# GradientBoostingClassifier 의 acc_score: [0.95614035 0.96491228 0.95614035 0.93859649 0.98230088]
# cross_val_score: 0.9596
# HistGradientBoostingClassifier 의 acc_score: [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088]
# cross_val_score: 0.9737
# KNeighborsClassifier 의 acc_score: [0.96491228 0.97368421 0.95614035 0.95614035 0.98230088]
# cross_val_score: 0.9666
# LabelPropagation 의 acc_score: [0.94736842 0.99122807 0.96491228 0.94736842 0.97345133]
# cross_val_score: 0.9649
# LabelSpreading 의 acc_score: [0.94736842 0.99122807 0.96491228 0.94736842 0.97345133]
# cross_val_score: 0.9649
# LinearDiscriminantAnalysis 의 acc_score: [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133]
# cross_val_score: 0.9614
# LinearSVC 의 acc_score: [0.97368421 1.         0.95614035 0.97368421 0.96460177]
# cross_val_score: 0.9736
# LogisticRegression 의 acc_score: [0.96491228 0.99122807 0.96491228 0.94736842 0.98230088]
# cross_val_score: 0.9701
# LogisticRegressionCV 의 acc_score: [0.98245614 1.         0.93859649 0.95614035 0.97345133]
# cross_val_score: 0.9701
# MLPClassifier 의 acc_score: [0.96491228 0.98245614 0.95614035 0.97368421 0.98230088]
# cross_val_score: 0.9719
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 acc_score: [0.85087719 0.88596491 0.78947368 0.81578947 0.84955752]
# cross_val_score: 0.8383
# NearestCentroid 의 acc_score: [0.93859649 0.95614035 0.92982456 0.92982456 0.94690265]
# cross_val_score: 0.9403
# NuSVC 의 acc_score: [0.94736842 0.95614035 0.95614035 0.93859649 0.95575221]
# cross_val_score: 0.9508
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 acc_score: [0.96491228 0.95614035 0.93859649 0.97368421 0.95575221]
# cross_val_score: 0.9578
# Perceptron 의 acc_score: [0.97368421 1.         0.94736842 0.97368421 0.96460177]
# cross_val_score: 0.9719
# QuadraticDiscriminantAnalysis 의 acc_score: [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265]
# cross_val_score: 0.9525
# RadiusNeighborsClassifier 의 acc_score: [       nan        nan 0.83333333 0.84210526        nan]
# cross_val_score: nan
# RandomForestClassifier 의 acc_score: [0.96491228 0.96491228 0.95614035 0.96491228 0.98230088]
# cross_val_score: 0.9666
# RidgeClassifier 의 acc_score: [0.94736842 0.97368421 0.93859649 0.93859649 0.95575221]
# cross_val_score: 0.9508
# RidgeClassifierCV 의 acc_score: [0.94736842 0.98245614 0.95614035 0.96491228 0.95575221]
# cross_val_score: 0.9613
# SGDClassifier 의 acc_score: [0.99122807 0.99122807 0.95614035 0.97368421 0.97345133]
# cross_val_score: 0.9771
# SVC 의 acc_score: [0.98245614 0.99122807 0.98245614 0.96491228 0.98230088]
# cross_val_score: 0.9807
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!