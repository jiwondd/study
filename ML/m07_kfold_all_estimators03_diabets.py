import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)

#2. 모델구성

allAlgorithms=all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        scores=cross_val_score(model,x,y,cv=kfold)
    
        print(name,'의 R2:',scores)
        print('cross_val_score:', round(np.mean(scores),4))
    except:
        # continue
        print(name,'은 안나온 놈!!!')
