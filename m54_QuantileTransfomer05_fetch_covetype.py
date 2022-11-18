from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
#                                 ㄴ이상치에 자유로운편
from sklearn.pipeline import make_pipeline
from icecream import ic
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1.데이터
datasets=fetch_covtype()
x,y=datasets.data,datasets.target
# ic(x.shape,y.shape)
# x.shape: (506, 13), y.shape: (506,)

le=LabelEncoder()
y=le.fit_transform(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=123)

mm=MinMaxScaler() 
stan=StandardScaler()
ma=MaxAbsScaler()
robus=RobustScaler() 
quan=QuantileTransformer()
power_yeo=PowerTransformer(method='yeo-johnson') 
power_box=PowerTransformer(method='box-cox')

scalers=[mm,stan,ma,robus,quan,power_yeo,power_box]
for scaler in scalers:
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        result=accuracy_score(y_test,y_predict)
        scale_name=scaler.__class__.__name__
        print('{0}결과:{1:4f}'.format(scale_name,result))
        
