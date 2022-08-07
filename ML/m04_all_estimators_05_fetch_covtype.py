
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=MinMaxScaler()
scaler=StandardScaler()
scaler=MaxAbsScaler()
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
        

# AdaBoostClassifier 의 정답률 :  0.4986876414550399            '--' 'c:\s
# BaggingClassifier 의 정답률 :  0.9612058208479988
# BernoulliNB 의 정답률 :  0.6592342710601276
# CalibratedClassifierCV 의 정답률 :  0.7111176131425179       
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 은 안나온 놈!!!
# DecisionTreeClassifier 의 정답률 :  0.9389086340283813       
# DummyClassifier 의 정답률 :  0.487698252196587
# ExtraTreeClassifier 의 정답률 :  0.8561482922127656
# ExtraTreesClassifier 의 정답률 :  0.952780909270845
# GaussianNB 의 정답률 :  0.09458447716496132
# GaussianProcessClassifier 은 안나온 놈!!!
# GradientBoostingClassifier 의 정답률 :  0.7706255432303813   
# HistGradientBoostingClassifier 의 정답률 :  0.7770969768422502
# KNeighborsClassifier 의 정답률 :  0.9273856957221415         2
# LabelPropagation 은 안나온 놈!!!
# LabelSpreading 은 안나온 놈!!!
# LinearDiscriminantAnalysis 의 정답률 :  0.67903582523687     
# LinearSVC 의 정답률 :  0.711212275070351
# LogisticRegression 의 정답률 :  0.7222963262566371
# LogisticRegressionCV 의 정답률 :  0.7234150581310294
# MLPClassifier 의 정답률 :  0.8773697753070059
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 은 안나온 놈!!!
# NearestCentroid 의 정답률 :  0.4493085376453276
# NuSVC 은 안나온 놈!!!
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.6002168618710361  
# Perceptron 의 정답률 :  0.5991497637754619
# QuadraticDiscriminantAnalysis 의 정답률 :  0.10395600802044698                                                            8
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.9554056263607652       
# RidgeClassifier 의 정답률 :  0.6995344354276568
# RidgeClassifierCV 의 정답률 :  0.6995430410574598
# SGDClassifier 의 정답률 :  0.7125375420600156
