import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

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

allAlgorithms=all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        model.fit(x_train,y_train)
    
        y_predict=model.predict(x_test)
        acc=accuracy_score(y_test,y_predict)
        print(name,'의 정답률 : ',acc)
    except:
        # continue
        print(name,'은 안나온 놈!!!')
        
# AdaBoostClassifier 의 정답률 :  0.8055555555555556
# BaggingClassifier 의 정답률 :  0.9722222222222222
# BernoulliNB 의 정답률 :  0.8333333333333334
# CalibratedClassifierCV 의 정답률 :  0.9722222222222222
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 은 안나온 놈!!!
# DecisionTreeClassifier 의 정답률 :  0.7777777777777778
# DummyClassifier 의 정답률 :  0.5
# ExtraTreeClassifier 의 정답률 :  0.7222222222222222
# ExtraTreesClassifier 의 정답률 :  1.0
# GaussianNB 의 정답률 :  0.9722222222222222
# GaussianProcessClassifier 의 정답률 :  0.9722222222222222
# GradientBoostingClassifier 의 정답률 :  0.9444444444444444
# HistGradientBoostingClassifier 의 정답률 :  1.0
# KNeighborsClassifier 의 정답률 :  0.8333333333333334
# LabelPropagation 의 정답률 :  0.8888888888888888
# LabelSpreading 의 정답률 :  0.8888888888888888
# LinearDiscriminantAnalysis 의 정답률 :  0.9722222222222222
# LinearSVC 의 정답률 :  0.9722222222222222
# LogisticRegression 의 정답률 :  0.9722222222222222
# LogisticRegressionCV 의 정답률 :  0.8888888888888888
# MLPClassifier 의 정답률 :  0.9722222222222222
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 은 안나온 놈!!!
# NearestCentroid 의 정답률 :  0.8888888888888888
# NuSVC 의 정답률 :  0.9444444444444444
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.9722222222222222
# Perceptron 의 정답률 :  0.9722222222222222
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9444444444444444
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  0.9722222222222222
# RidgeClassifierCV 의 정답률 :  0.9722222222222222
# SGDClassifier 의 정답률 :  1.0
# SVC 의 정답률 :  1.0
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!