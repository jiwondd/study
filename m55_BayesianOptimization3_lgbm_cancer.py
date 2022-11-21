from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets=load_breast_cancer()
x,y=datasets.data,datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=123,train_size=0.8)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

bayesian_params={
    'max_depth':(6,16),
    'num_leaves':(24,64),
    'min_child_samples':(10,200),
    'min_child_weight':(1,50),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'max_bin':(10,500),
    'reg_lambda':(0.001,10),
    'reg_alpha':(0.01,50)   
}

def lgb_hamsu(max_depth,num_leaves,min_child_samples,min_child_weight,
              subsample,colsample_bytree,max_bin,reg_lambda,reg_alpha):
    params={
        'n_estimators':500,"learning_rate":0.02,
        'max_depth':int(round(max_depth)), #정수로
        'num_leaves':int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'subsample':max(min(subsample,1),0), # 0~1 사이로
        'colsample_bytree':max(min(colsample_bytree,1),0),
        'max_bin':max(int(round(max_bin)),10), # 10이상의 정수
        'reg_lambda':max(reg_lambda,0), #양수만 받는다 0이상
        'reg_alpha':max(reg_alpha,0)
    }
    
    model=LGBMClassifier(**params) # **키워드 받을게(딕셔너리형태) *여러개의 인자를 받을게(1개도되고 여러개도되고)
    model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='accuracy_score',
              verbose=0,
              early_stopping_rounds=50)
    y_pred=model.predict(x_test)
    result=accuracy_score(y_test,y_pred)
    
    return result

lgb_bo=BayesianOptimization(f=lgb_hamsu,
                            pbounds=bayesian_params,
                            random_state=123)

lgb_bo.maximize(init_points=5,n_iter=50)

print(lgb_bo.max)
    
    
