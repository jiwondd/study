from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from icecream import ic
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# 1.데이터
datasets=load_digits()
x,y=datasets.data,datasets.target
# ic(x.shape,y.shape)

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

# ic| x.shape: (1797, 64)
# ic| score: 0.8666666666666667
# ic| cv_score: array([0.84722222, 0.85416667, 0.87804878, 0.85017422, 0.87804878])
#     np.mean(cv_score): 0.8615321331784747
    
# ic| xp.shape: (1797, 2145)
# ic| poly_score: 0.85
# ic| cv_score2: array([0.83333333, 0.83680556, 0.83972125, 0.85365854, 0.83972125])
#     np.mean(cv_score2): 0.8406479868370113