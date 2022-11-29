from sklearn.datasets import fetch_california_housing
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
datasets=fetch_california_housing()
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
xp=np.delete(xp,[0],axis=1) 

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

# ic| score: 0.6065722122106435
# ic| cv_score: array([0.11678787, 0.61520602, 0.57715973, 0.61498804, 0.59887989])
#     np.mean(cv_score): 0.5046043060069918

# ic| poly_score: 0.5005165687198212
# ic| cv_score2: array([-9.60459459,  0.6936112 ,  0.46502495,  0.68331427,  0.67358888])
#     np.mean(cv_score2): -1.4178110591460573

# bias 1 지워봄
# ic| poly_score: 0.5005165687196428
# ic| cv_score2: array([-9.60459459,  0.6936112 ,  0.46502495,  0.68331427,  0.67358888])
#     np.mean(cv_score2): -1.4178110591469915
