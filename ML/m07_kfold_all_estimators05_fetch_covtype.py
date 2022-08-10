
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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets=fetch_covtype()
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