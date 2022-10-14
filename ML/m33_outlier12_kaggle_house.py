import numpy as np
import datetime as dt 
import pandas as pd
from collections import Counter
import datetime as dt
from sqlalchemy import asc
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#.1 데이터
path='./_data/kaggle_house/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!

numerical_feats=train_set.dtypes[train_set.dtypes !='object'].index
categorial_feats=train_set.dtypes[train_set.dtypes =='object'].index

# print(train_set.isnull().sum())
# print(test_set.isnull().sum())

def getZscoreOutlier(df,col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:",out)
    print("min",np.min(out))
    return np.min(out)

col = "LotFrontage"
minOutlier = getZscoreOutlier(train_set,col)
print(train_set[train_set[col] >= minOutlier])



