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
import warnings

warnings.filterwarnings('ignore')

#1. 데이터 
datasets=load_iris()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
allAlgorithms=all_estimators(type_filter='classifier')
# print('모델의 갯수:',len(allAlgorithms)) 모델의 갯수: 41
# print('allAlgorithms',allAlgorithms)  딕셔너리=[키,밸류]형태로 되어잇음
# allAlgorithms [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), 
# ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), 
# ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), 
# ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), 
# ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), 
# ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), 
# ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), 
# ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), 
# ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), 
# ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), 
# ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>), ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), 
# ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), 
# ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), ('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>), 
# ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), 
# ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), 
# ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>),
# ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), 
# ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), 
# ('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), ('SVC', <class 'sklearn.svm._classes.SVC'>), ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), 
# ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]

# for (name, algorithm) in allAlgorithms:
#     model=algorithm()
#     model.fit(x_train,y_train)
    
#     y_predict=model.predict(x_test)
#     acc=accuracy_score(y_test,y_predict)
#     print(name,'의 정답률 : ',acc)
    
# AdaBoostClassifier 의 정답률 :  0.9
# BaggingClassifier 의 정답률 :  0.9
# BernoulliNB 의 정답률 :  0.4
# CalibratedClassifierCV 의 정답률 :  0.7666666666666667
# CategoricalNB 의 정답률 :  0.3 
# TypeError: __init__() missing 1 required positional argument: 'base_estimator' <-버전문제로 안돌아가는 애들이 있다. 그니까 *예외처리를 배워보자*


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
    
    
    
    
    
    
    
    
# AdaBoostClassifier 의 정답률 :  0.9
# BaggingClassifier 의 정답률 :  0.9333333333333333
# BernoulliNB 의 정답률 :  0.4
# CalibratedClassifierCV 의 정답률 :  0.7666666666666667
# CategoricalNB 의 정답률 :  0.3
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 :  0.6
# DecisionTreeClassifier 의 정답률 :  0.9333333333333333
# DummyClassifier 의 정답률 :  0.26666666666666666
# ExtraTreeClassifier 의 정답률 :  0.8666666666666667
# ExtraTreesClassifier 의 정답률 :  0.9333333333333333
# GaussianNB 의 정답률 :  0.9666666666666667
# GaussianProcessClassifier 의 정답률 :  0.9
# GradientBoostingClassifier 의 정답률 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.9333333333333333
# KNeighborsClassifier 의 정답률 :  0.9333333333333333
# LabelPropagation 의 정답률 :  0.9666666666666667
# LabelSpreading 의 정답률 :  0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667
# LinearSVC 의 정답률 :  0.8
# LogisticRegression 의 정답률 :  0.9
# LogisticRegressionCV 의 정답률 :  0.9666666666666667
# MLPClassifier 의 정답률 :  0.8666666666666667
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 :  0.6
# NearestCentroid 의 정답률 :  0.9333333333333333
# NuSVC 의 정답률 :  0.9666666666666667
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.8333333333333334
# Perceptron 의 정답률 :  0.6333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
# RadiusNeighborsClassifier 의 정답률 :  0.6
# RandomForestClassifier 의 정답률 :  0.9666666666666667
# RidgeClassifier 의 정답률 :  0.6666666666666666
# RidgeClassifierCV 의 정답률 :  0.6666666666666666
# SGDClassifier 의 정답률 :  0.7
# SVC 의 정답률 :  0.9666666666666667
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
# 34개의 모델이 돌아감