from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
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
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

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

train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','MonthlyIncome','NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','MonthlyIncome','NumberOfFollowups'], axis=1)
train_set['TypeofContact'].fillna('N', inplace=True)
test_set['TypeofContact'].fillna('N', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)

total_set = pd.get_dummies(total_set)

imputer=IterativeImputer(random_state=123)
imputer.fit(total_set)
total_set=imputer.transform(total_set)

scaler=QuantileTransformer()
scaler.fit(total_set)
x=scaler.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=123,train_size=0.8,stratify=y)

kFold=StratifiedKFold(shuffle=True,random_state=123)

smote=SMOTE(random_state=123)
x_train,y_train=smote.fit_resample(x_train,y_train)
# print(np.unique(y, return_counts=True))

# 2. 모델구성
# model=RandomForestClassifier(random_state=123)
# model=CatBoostClassifier(random_state=123)
model=XGBClassifier(random_state=123)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
# print('model.score:',result) 

#5. 데이터 summit
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
print('model.score:',result) 
submission.to_csv('./_data/dacon_travel/sample_submission_smote.csv', index=True)


# model.score: 0.8695652173913043 RF
# model.score: 0.8414322250639387 xgb
# model.score: 0.9258312020460358 RF smote
# model.score: 0.8849104859335039 
# model.score: 0.9332273449920508 <-트레인테스트전에 스모트
# model.score: 0.9411764705882353
# model.score: 0.9395866454689984 <- 'MonthlyIncome'
# model.score: 0.9411764705882353 <- 'NumberOfFollowups'
# model.score: 0.8746803069053708 <- 스케일러 전체셋에 하니까 점수가 맞음
# model.score: 0.8772378516624041 <- 'MonthlyIncome','NumberOfFollowups' 다시 살림
# model.score: 0.8849104859335039 <- 스모트 빼버림
# model.score: 0.8849104859335039 <- xg
# model.score: 0.8823529411764706 <- 다시 스모트 넣기