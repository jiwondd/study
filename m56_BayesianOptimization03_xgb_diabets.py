from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets=load_diabetes()
x,y=datasets.data,datasets.target

# le=LabelEncoder()
# y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=123,train_size=0.8)

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



