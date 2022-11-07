from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from icecream import ic
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# 1.데이터
datasets=load_boston()
x,y=datasets.data,datasets.target
# ic(x.shape,y.shape)
# x.shape: (506, 13), y.shape: (506,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=KFold(n_splits=5,shuffle=True,random_state=123)

# 2.모델
model=make_pipeline(StandardScaler(),
                    LinearRegression()
                    )

model.fit(x_train,y_train)

score=model.score(x_test,y_test)
cv_score=cross_val_score(model,x_train, y_train, cv=kfold,scoring='r2')
ic(score) #score: 0.7665382927362877 (poly적용전)
ic(cv_score,np.mean(cv_score))

############polynomial적용후###############
pf=PolynomialFeatures(degree=2) #include_bias=False
xp=pf.fit_transform(x)
# print(xp.shape) #(506, 105)

x_train2,x_test2,y_train2,y_test2=train_test_split(xp,y,
        train_size=0.8,shuffle=True, random_state=1234)

# 2.모델
model2=make_pipeline(StandardScaler(),
                    LinearRegression()
                    )

model.fit(x_train2,y_train2)

poly_score=model.score(x_test2,y_test2)
cv_score2=cross_val_score(model2,x_train2, y_train2, cv=kfold,scoring='r2')
ic(poly_score) #score: 0.8745129304823863 (poly적용후)
ic(cv_score2,np.mean(cv_score2))

# score: 0.7665382927362877
# cv_score: array([0.70659128, 0.68526485, 0.74370188, 0.64025164, 0.6334187 ])   
# np.mean(cv_score): 0.6818456717897765

# poly_score: 0.8745129304823863
# cv_score2: array([0.13014823, 0.72676279, 0.79752314, 0.70671284, 0.74682703])  
# np.mean(cv_score2): 0.6215948083143407