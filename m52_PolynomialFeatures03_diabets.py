from sklearn.datasets import load_diabetes
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
datasets=load_diabetes()
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
xp=np.delete(xp,[0],axis=1) 
# ic(xp.shape)
# ic(xp)

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

# ic| score: 0.46263830098374936
# ic| cv_score: array([0.51583902, 0.46958492, 0.38394311, 0.54405474, 0.56107605])
#     np.mean(cv_score): 0.4948995654862614

# ic| poly_score: 0.4226086711284325
# ic| cv_score2: array([ 0.41215695, -2.82578882,  0.1802958 ,  0.35961316,  0.50029688])
#     np.mean(cv_score2): -0.2746852054585986

# 마이너스 나오길래 0번째 (bias 1 나오는거 지워봄)
# ic| poly_score: 0.4186731903894866
# ic| cv_score2: array([0.39529337, 0.34631983, 0.15605019, 0.35399376, 0.50029688])
#     np.mean(cv_score2): 0.3503908050674226