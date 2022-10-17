# https://dacon.io/competitions/official/235959/overview/description

from cProfile import label
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

#.1 데이터
path='./_data/dacon_travel/'
train=pd.read_csv(path+'train.csv')
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!

# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)

# 결측치 평균으로 채우기
train_nona = train.copy()
mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome']
for col in mean_cols:
    train_nona[col] = train_nona[col].fillna(train[col].mean())

# 문자형 변수 바꿔주기
object_columns = train.columns[train.dtypes == 'object']
# print('object 칼럼은 다음과 같습니다 : ', list(object_columns))
train_enc = train_nona.copy()
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])
    
# 숫자형 변수 스케일링
scaler = MinMaxScaler()
train_scale = train_enc.copy()
scaler.fit(train_scale[['Age', 'DurationOfPitch', 'MonthlyIncome']])
train_scale[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(train_scale[['Age', 'DurationOfPitch', 'MonthlyIncome']])

# 위 과정 test셋 적용
mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome']
for col in mean_cols:
    test[col] = test[col].fillna(test[col].mean())

for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_nona[o_col])
    test[o_col] = encoder.transform(test[o_col])
    
test[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test[['Age', 'DurationOfPitch', 'MonthlyIncome']])

model = RandomForestClassifier()
train = train_scale.drop(columns=['id'])
test = test.drop(columns=['id'])

x_train = train.drop(columns=['ProdTaken'])
y_train = train[['ProdTaken']]

model.fit(x_train,y_train)

prediction = model.predict(test)

#5. 데이터 summit
y_summit = model.predict(test)
submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/dacon_travel/sample_submission2.csv', index=True)

