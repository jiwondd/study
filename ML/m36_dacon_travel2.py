# https://dacon.io/competitions/official/235959/overview/description

from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


#.1 데이터
path='./_data/dacon_travel/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!

# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)
train_set = train_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar','NumberOfTrips'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar','NumberOfTrips'], axis=1)
train_set['TypeofContact'].fillna('N', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)

le = LabelEncoder()
cols = ('TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(total_set[c].values)) 
    total_set[c] = lbl.transform(list(total_set[c].values))

print(total_set.head())

imputer=IterativeImputer(random_state=42)
imputer.fit(total_set)
total_set=imputer.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

# x=pd.DataFrame(x)
# print(x.isnull().sum()) 
# 임퓨터 제대로 들어갔는지 보려고 넘파이를 데이터프레임으로 잠깐 바꿔봄

scaler=MinMaxScaler()
scaler.fit(train_set)
x['Age']=scaler.fit_transform(x['Age'])

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=123,train_size=0.9,stratify=y)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=123)


# 2. 모델구성
model=RandomForestClassifier(random_state=123)
# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 
# best_params=model.best_params_
# print('best_params : ', best_params )

#5. 데이터 summit
# y_summit = model.predict(test_set)
# submission['ProdTaken'] = y_summit
# print(submission)
# submission.to_csv('./_data/dacon_travel/sample_submission.csv', index=True)


# model.score: 0.8695652173913043 RF
# model.score: 0.8414322250639387 xgb
# model.score: 0.8877551020408163
# model.score: 0.8877551020408163