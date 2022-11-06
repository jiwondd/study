
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from icecream import ic
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

#1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets.target

# x = np.delete(x,[1,3,5],axis=1)
# print(x.shape) #(581012, 54)
le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

ic(x.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234,stratify=y)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=123)

# 2.모델
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model=make_pipeline(StandardScaler(),
                    DecisionTreeClassifier()
                    )

model.fit(x_train,y_train)

score=model.score(x_test,y_test)
cv_score=cross_val_score(model,x_train, y_train, cv=kfold)
ic(score) 
ic(cv_score,np.mean(cv_score))

############polynomial적용후###############
pf=PolynomialFeatures(degree=2) #include_bias=False
xp=pf.fit_transform(x)
ic(xp.shape)
# ic(xp)
# xp=np.delete(xp,[0],axis=1) 
# ic(xp.shape)
# ic(xp)

x_train2,x_test2,y_train2,y_test2=train_test_split(xp,y,
        train_size=0.8,shuffle=True, random_state=1234,stratify=y)

# 2.모델
model2=make_pipeline(StandardScaler(),
                    DecisionTreeClassifier()
                    )

model.fit(x_train2,y_train2)

poly_score=model.score(x_test2,y_test2)
cv_score2=cross_val_score(model2,x_train2, y_train2, cv=kfold)
ic(poly_score)
ic(cv_score2,np.mean(cv_score2))

# ic| x.shape: (581012, 54)
# ic| score: 0.9392528592205021
# ic| cv_score: array([0.93141284, 0.93183236, 0.9323487 , 0.93218735, 0.93378944])
#     np.mean(cv_score): 0.9323141365061822

# ic| xp.shape: (581012, 1540)
# ic| poly_score: 0.937712451485762
# ic| cv_score2: array([0.92894946, 0.92797057, 0.93000366, 0.92993911, 0.9292499 ])
#     np.mean(cv_score2): 0.9292225409304888