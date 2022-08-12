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

#.1 데이터
path='./_data/dacon_travel/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!

# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)

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
# print(total_set.info())
# print(total_set.isnull().sum())

total_set.loc[total_set['Age'] != total_set['Age'], 'Age'] = total_set['Age'].median()
total_set.loc[total_set['DurationOfPitch'] != total_set['DurationOfPitch'], 'DurationOfPitch'] = total_set['DurationOfPitch'].median()
total_set.loc[total_set['NumberOfFollowups'] != total_set['NumberOfFollowups'], 'NumberOfFollowups'] = total_set['NumberOfFollowups'].median()
total_set.loc[total_set['PreferredPropertyStar'] != total_set['PreferredPropertyStar'], 'PreferredPropertyStar'] = total_set['PreferredPropertyStar'].median()
total_set.loc[total_set['NumberOfTrips'] != total_set['NumberOfTrips'], 'NumberOfTrips'] = total_set['NumberOfTrips'].median()
total_set.loc[total_set['NumberOfChildrenVisiting'] != total_set['NumberOfChildrenVisiting'], 'NumberOfChildrenVisiting'] = total_set['NumberOfChildrenVisiting'].median()
total_set.loc[total_set['MonthlyIncome'] != total_set['MonthlyIncome'], 'MonthlyIncome'] = total_set['MonthlyIncome'].median()
print(total_set.isnull().sum())

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

lda=LinearDiscriminantAnalysis() 
lda.fit(x,y)
x=lda.transform(x)

print(x.shape)
print(x)
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

# lda=LinearDiscriminantAnalysis() 
# lda.fit(x,y)
# x=lda.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=123,train_size=0.8,stratify=y)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=123)

prameters={'n_estimator':[100,200,300,],'learnig_rate':[0.1,0.2,0.01,0.001]},{'max_depth':[None,2,3,4,5],'gamma':[0,1,2,3]},{'reg_alpha':[0.1,0,0.001,1],'reg_lambda':[0.1,0,0.001,1]}

# 2. 모델구성
xgb=XGBClassifier(random_state=123,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
model=GridSearchCV(xgb,prameters,cv=kFold,n_jobs=8)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

#5. 데이터 summit
y_summit = model.predict(test_set)
y_summit = y_summit.flatten()                 
y_summit = np.where(y_summit > 0.55, 1 , 0)   

submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/dacon_travel/sample_submission.csv', index=True)

