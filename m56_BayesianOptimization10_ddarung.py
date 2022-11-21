import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from icecream import ic
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count','hour_bef_precipitation','hour_bef_humidity'],axis=1)
y=train_set['count']

# print(x.shape) #(1328, 9)->(1328, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=1234)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

bayesian_params={
                 'max_depth':[1,6],
                 'min_child_weight':[0.1,5],
                 'reg_alpha':[0,10],
                 'reg_lambda':[0,10]
                }

def lgb_hamsu(max_depth,min_child_weight,
              reg_lambda,reg_alpha):
    params={
             'n_estimators':200,"learning_rate":0.02,
             'max_depth':int(round(max_depth)),
             'min_child_weight':int(round(min_child_weight)),
             'reg_lambda':max(reg_lambda,0),
             'reg_alpha':max(reg_alpha,0)
    }
    
    model=XGBRegressor(**params) # **키워드 받을게(딕셔너리형태) *여러개의 인자를 받을게(1개도되고 여러개도되고)
    model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50)
    y_pred=model.predict(x_test)
    result=r2_score(y_test,y_pred)
    
    return result

lgb_bo=BayesianOptimization(f=lgb_hamsu,
                            pbounds=bayesian_params,
                            random_state=123)

lgb_bo.maximize(init_points=2,n_iter=20)

print(lgb_bo.max)