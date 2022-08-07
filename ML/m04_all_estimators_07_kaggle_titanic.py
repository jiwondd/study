import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# 1. 데이터
path='./_data/kaggle_titanic/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv')
train = train_set.drop(['PassengerId','Name', 'Ticket','Cabin'], axis = 1 )
test = test_set.drop(['Name', 'Ticket','Cabin'], axis= 1)

sex_train_dummies = pd.get_dummies(train['Sex'])
sex_test_dummies = pd.get_dummies(test['Sex'])

sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']

train.drop(['Sex'], axis=1, inplace=True)
test.drop(['Sex'], axis=1, inplace=True)

train = train.join(sex_train_dummies)
test = test.join(sex_test_dummies)

train["Age"].fillna(train["Age"].mean() , inplace=True)
test["Age"].fillna(train["Age"].mean() , inplace=True)

train["Embarked"].fillna('S', inplace=True)
test["Embarked"].fillna('S', inplace=True)

embarked_train_dummies = pd.get_dummies(train['Embarked'])
embarked_test_dummies = pd.get_dummies(test['Embarked'])

embarked_train_dummies.columns = ['S', 'C', 'Q']
embarked_test_dummies.columns = ['S', 'C', 'Q']

train.drop(['Embarked'], axis=1, inplace=True)
test.drop(['Embarked'], axis=1, inplace=True)

train = train.join(embarked_train_dummies)
test = test.join(embarked_test_dummies)

X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,
        train_size=0.8,shuffle=True, random_state=31)

# 2. 모델구성
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
        
# AdaBoostClassifier 의 정답률 :  0.8268156424581006
# BaggingClassifier 의 정답률 :  0.776536312849162
# BernoulliNB 의 정답률 :  0.8044692737430168
# CalibratedClassifierCV 의 정답률 :  0.7932960893854749
# CategoricalNB 의 정답률 :  0.7988826815642458
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 :  0.6703910614525139
# DecisionTreeClassifier 의 정답률 :  0.776536312849162
# DummyClassifier 의 정답률 :  0.553072625698324
# ExtraTreeClassifier 의 정답률 :  0.7988826815642458
# ExtraTreesClassifier 의 정답률 :  0.8044692737430168
# GaussianNB 의 정답률 :  0.7821229050279329
# GaussianProcessClassifier 의 정답률 :  0.6983240223463687
# GradientBoostingClassifier 의 정답률 :  0.8156424581005587
# HistGradientBoostingClassifier 의 정답률 :  0.8100558659217877
# KNeighborsClassifier 의 정답률 :  0.6759776536312849
# LabelPropagation 의 정답률 :  0.6871508379888268
# LabelSpreading 의 정답률 :  0.6815642458100558
# LinearDiscriminantAnalysis 의 정답률 :  0.7932960893854749
# LinearSVC 의 정답률 :  0.6759776536312849
# LogisticRegression 의 정답률 :  0.7821229050279329
# LogisticRegressionCV 의 정답률 :  0.776536312849162
# MLPClassifier 의 정답률 :  0.8268156424581006
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 :  0.6759776536312849
# NearestCentroid 의 정답률 :  0.5921787709497207
# NuSVC 의 정답률 :  0.8044692737430168
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 :  0.776536312849162
# Perceptron 의 정답률 :  0.7318435754189944
# QuadraticDiscriminantAnalysis 의 정답률 :  0.7541899441340782
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 :  0.7821229050279329
# RidgeClassifier 의 정답률 :  0.7932960893854749
# RidgeClassifierCV 의 정답률 :  0.7932960893854749
# SGDClassifier 의 정답률 :  0.7541899441340782
# SVC 의 정답률 :  0.5977653631284916
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!