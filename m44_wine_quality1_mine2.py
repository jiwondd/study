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
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
#.1 데이터
path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=0, sep=';')
# print(data_set.shape) (4898, 11)                             ㄴ기준으로 컬럼을 나눠줘

#.1 데이터
y=data_set['quality']
x=data_set.drop(['quality'],axis=1)


le=LabelEncoder()
y=le.fit_transform(y)

scaler=MinMaxScaler()
scaler.fit(x)
data_set=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,
                                                  random_state=123,shuffle=True)

kFold=StratifiedKFold(n_splits=2, shuffle=True,random_state=123)

XG_parameters={'n_estimators':[100],
            'learning_rate':[0.001],
            'max_depth':[3],
            'gamma':[0],
            'min_child_weight':[1],
            'subsample':[0.1],
            'colsample_bytree':[0],
            'colsample_bylevel':[0],
            'colsample_bynode':[0],
            'reg_alpha':[2],
            'reg_lambda':[2]
           }

RF_parameters = [
    {'n_estimators':[100,200,300]},
    {'max_depth':[1,3,6,8,10]},
    {'min_samples_leaf':[4,8,12]},
    {'min_samples_split':[2,4,6,8]},
    {'max_depth' : [6, 8, 10, 12]}
]

# 2. 모델구성
# XG =XGBClassifier(random_state=123)
# RF = RandomForestClassifier(random_state=123)
# model=GridSearchCV(RF,RF_parameters,cv=kFold,n_jobs=8)
model=RandomForestClassifier(random_state=123,
                    n_estimators=100,
                    max_depth=6)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

print(model.feature_importances_)
thresholds=model.feature_importances_
print('---------------------------------')

for thresh in thresholds :
    selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train=selection.transform(x_train)
    select_x_test=selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)
    
    selection_model=XGBClassifier(n_jobs=-1,
                                 random_state=100,
                                 n_estimators=100,
                                 learning_rate=0.3,
                                 max_depth=6,
                                 gamma=0)
    selection_model.fit(select_x_train,y_train)
    y_predict=selection_model.predict(select_x_test)
    score=accuracy_score(y_test,y_predict)
    print("Thresh=%.3f,n=%d, acc:%.2f%%"
          #소수점3개까지,정수,소수점2개까지
          %(thresh,select_x_train.shape[1],score*100))

