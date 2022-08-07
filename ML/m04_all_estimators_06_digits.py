from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical
import sklearn as sk


#1. 데이터
datasets=load_digits()
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
        
# AdaBoostClassifier 의 정답률 :  0.46111111111111114
# BaggingClassifier 의 정답률 :  0.925       
# BernoulliNB 의 정답률 :  0.8583333333333333
# CalibratedClassifierCV 의 정답률 :  0.9611111111111111
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 은 안나온 놈!!!
# DecisionTreeClassifier 의 정답률 :  0.8472222222222222
# DummyClassifier 의 정답률 :  0.075
# ExtraTreeClassifier 의 정답률 :  0.7194444444444444   
# ExtraTreesClassifier 의 정답률 :  0.9805555555555555
# GaussianNB 의 정답률 :  0.8611111111111112
# GaussianProcessClassifier 의 정답률 :  0.9444444444444444
# GradientBoostingClassifier 의 정답률 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.975   
# KNeighborsClassifier 의 정답률 :  0.9194444444444444
# LabelPropagation 의 정답률 :  0.9277777777777778
# LabelSpreading 의 정답률 :  0.9277777777777778
# LinearDiscriminantAnalysis 의 정답률 :  0.95  
# LinearSVC 의 정답률 :  0.9666666666666667
# LogisticRegression 의 정답률 :  0.9611111111111111
# LogisticRegressionCV 의 정답률 :  0.9638888888888889
# MLPClassifier 의 정답률 :  0.975
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 은 안나온 놈!!!
# NearestCentroid 의 정답률 :  0.7444444444444445
# NuSVC 의 정답률 :  0.9166666666666666
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.9444444444444444
# Perceptron 의 정답률 :  0.9361111111111111
# QuadraticDiscriminantAnalysis 의 정답률 :  0.8666666666666667
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.9777777777777777
# RidgeClassifier 의 정답률 :  0.9166666666666666
# RidgeClassifierCV 의 정답률 :  0.9222222222222223
# SGDClassifier 의 정답률 :  0.95
# SVC 의 정답률 :  0.9666666666666667
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!