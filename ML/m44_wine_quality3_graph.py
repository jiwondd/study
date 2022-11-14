import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from cProfile import label
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

#.1 데이터
path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=None, sep=';')
# print(data_set.shape) (4898, 11)                                ㄴ기준으로 컬럼을 나눠줘

print(data_set.describe())
print(data_set.info())

y=data_set['quality']
x=data_set.drop(['quality'],axis=1)

#########그래프 그리기#########
# 1. value_counts 쓰지마라
# 2. groupby , count() 써보기
# plt.bar (quality)

import matplotlib.pyplot as plt
count_data = data_set.groupby('quality')['quality'].count()
print(count_data)
plt.bar(count_data.index, count_data)
plt.show()
