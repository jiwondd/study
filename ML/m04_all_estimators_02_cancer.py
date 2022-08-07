import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

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

# AdaBoostClassifier 의 정답률 :  0.9473684210526315
# BaggingClassifier 의 정답률 :  0.9385964912280702
# BernoulliNB 의 정답률 :  0.9035087719298246
# CalibratedClassifierCV 의 정답률 :  0.9473684210526315
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 은 안나온 놈!!!
# DecisionTreeClassifier 의 정답률 :  0.8947368421052632
# DummyClassifier 의 정답률 :  0.6666666666666666
# ExtraTreeClassifier 의 정답률 :  0.9298245614035088
# ExtraTreesClassifier 의 정답률 :  0.9736842105263158
# GaussianNB 의 정답률 :  0.9210526315789473
# GaussianProcessClassifier 의 정답률 :  0.9649122807017544
# GradientBoostingClassifier 의 정답률 :  0.9298245614035088
# HistGradientBoostingClassifier 의 정답률 :  0.9385964912280702
# KNeighborsClassifier 의 정답률 :  0.9649122807017544
# LabelPropagation 의 정답률 :  0.9385964912280702
# LabelSpreading 의 정답률 :  0.9385964912280702
# LinearDiscriminantAnalysis 의 정답률 :  0.9385964912280702
# LinearSVC 의 정답률 :  0.956140350877193
# LogisticRegression 의 정답률 :  0.9649122807017544
# LogisticRegressionCV 의 정답률 :  0.9649122807017544
# MLPClassifier 의 정답률 :  0.9473684210526315
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 은 안나온 놈!!!
# NearestCentroid 의 정답률 :  0.9210526315789473
# NuSVC 의 정답률 :  0.9385964912280702
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.9649122807017544
# Perceptron 의 정답률 :  0.9473684210526315
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.9473684210526315
# RidgeClassifier 의 정답률 :  0.9473684210526315
# RidgeClassifierCV 의 정답률 :  0.9473684210526315
# SGDClassifier 의 정답률 :  0.9473684210526315
# SVC 의 정답률 :  0.956140350877193
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!