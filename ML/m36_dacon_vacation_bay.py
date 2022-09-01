from cProfile import label
from tkinter import Y
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier 
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score

#.1 데이터
path='./_data/dacon_travel/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)

train_set = train_set.replace({'Gender' : 'Fe Male'}, 'Female')
test_set = test_set.replace({'Gender' : 'Fe Male'}, 'Female')
train_set = train_set.replace({'Occupation':'Free Lancer'}, 'Small Business')
test_set = test_set.replace({'Occupation':'Free Lancer'}, 'Small Business')
train_set = train_set.replace({'MaritalStatus' : 'Divorced'}, 'Single')
test_set = test_set.replace({'MaritalStatus' : 'Divorced'}, 'Single')

train_set.loc[train_set['NumberOfTrips'] != train_set['NumberOfTrips'], 'NumberOfTrips'] = train_set['NumberOfTrips'].fillna(0)
test_set.loc[test_set['NumberOfTrips'] != test_set['NumberOfTrips'], 'NumberOfTrips'] = test_set['NumberOfTrips'].fillna(0)
train_set.loc[train_set['DurationOfPitch'] != train_set['DurationOfPitch'], 'DurationOfPitch'] = train_set['DurationOfPitch'].fillna(0)
test_set.loc[test_set['DurationOfPitch'] != test_set['DurationOfPitch'], 'DurationOfPitch'] = test_set['DurationOfPitch'].fillna(0)

train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)
train_set['DurationOfPitch']=np.round(train_set['DurationOfPitch'],0).astype(int)
test_set['DurationOfPitch']=np.round(test_set['DurationOfPitch'],0).astype(int)
train_set['MonthlyIncome']=np.round(train_set['MonthlyIncome'],0).astype(int)
test_set['MonthlyIncome']=np.round(test_set['MonthlyIncome'],0).astype(int)

train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','NumberOfFollowups'], axis=1)
train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)
# print(total_set.head(10))

total_set = pd.get_dummies(total_set)
# print(total_set.isnull().sum())

# imputer=IterativeImputer(random_state=777)
# imputer.fit(total_set)
# total_set=imputer.transform(total_set)

scaler=QuantileTransformer()
# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(total_set)
x=scaler.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=777,train_size=0.8,stratify=y)

kFold=StratifiedKFold(shuffle=True,random_state=777)

smote=SMOTE(random_state=777)
x,y=smote.fit_resample(x_train,y_train)
# print(np.unique(y, return_counts=True))

# 2. 모델구성
# model=RandomForestClassifier(random_state=777)
# model=CatBoostClassifier(random_state=777)
# model=XGBClassifier(random_state=777)

cat_paramets = {"learning_rate" : [0.20909079092170735],
                'depth' : [8],
                'od_pval' : [0.236844398775451],
                'model_size_reg': [0.30614059763442997],
                'l2_leaf_reg' :[5.535171839105427]}

cat = CatBoostClassifier(random_state=123,verbose=False,n_estimators=500)
model = RandomizedSearchCV(cat,cat_paramets,cv=kFold,n_jobs=-1)


# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
# print('model.score:',result) 

#5. 데이터 summit
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
print('model.score:',result) 
# submission.to_csv('./_data/dacon_travel/sample_submission_fill0.csv', index=True)

# model.score: 0.8925831202046036
# model.score: 0.8951406649616368 <-전처리 추가 (median)
# model.score: 0.8976982097186701 <- mean
# model.score: 0.9028132992327366 <-fillna(0)
# model.score: 0.9028132992327366 <-스탠다드 스케일
# model.score: 0.9028132992327366 <-RobustScaler