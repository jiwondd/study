from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
#                                 ㄴ이상치에 자유로운편
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from icecream import ic
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,accuracy_score

# 1.데이터
datasets=fetch_california_housing()
x,y=datasets.data,datasets.target
# ic(x.shape,y.shape)
# x.shape: (506, 13), y.shape: (506,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=KFold(n_splits=5,shuffle=True,random_state=123)

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
        model=RandomForestRegressor()
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        result=r2_score(y_test,y_predict)
        scale_name=scaler.__class__.__name__
        print('{0}결과:{1:4f}'.format(scale_name,result))
        
# MinMaxScaler결과:0.802980
# StandardScaler결과:0.803962
# MaxAbsScaler결과:0.804237
# RobustScaler결과:0.804199
# QuantileTransformer결과:0.805501
# PowerTransformer결과:0.804500
# The Box-Cox transformation can only be applied to strictly positive data