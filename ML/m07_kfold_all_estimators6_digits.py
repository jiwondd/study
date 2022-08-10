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
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets=load_digits()
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