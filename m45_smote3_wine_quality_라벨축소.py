# 라벨값 축소하기
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
from imblearn.over_sampling import SMOTE

#.1 데이터
path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=None, sep=';')
# print(data_set.shape) (4898, 11)                                ㄴ기준으로 컬럼을 나눠줘

# print(data_set.describe())
# print(data_set.info())

# y=data_set2[:,11] #np일때 x,y나누기

y=data_set['quality']
x=data_set.drop(['quality'],axis=1) #df일때 x,y나누기

newlist=[]
for i in y:
    if i <=4:
        newlist+=[0]
    elif i <=6:
        newlist+=[1]
    else:
        newlist+=[2]
        
# print(np.unique(newlist,return_counts=True))
# (array([0, 1, 2]), array([ 183, 3655, 1060], dtype=int64))

scaler=MinMaxScaler()
scaler.fit(x)
data_set=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,newlist,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

# 2. 모델구성
model = RandomForestClassifier(random_state=123)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score(macro)',f1_score(y_test,y_pred, average='macro'))
print(pd.Series(y_train).value_counts()) #[146]
print('==========SOMTE_적용==========')
smote=SMOTE(random_state=123, k_neighbors=146)
x_train,y_train=smote.fit_resample(x_train,y_train)
# print(pd.Series(y_train).value_counts()) #[2924]

model=RandomForestClassifier() 
model.fit(x_train,y_train)

# test데이터는(평가데이터_) 스케일 하면 안됨 건들지말자

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score(macro)',f1_score(y_test,y_pred, average='macro'))

# model.score: 0.8765306122448979
# acc_score : 0.8765306122448979
# f1_score(macro) 0.6652651082750519
# ==========SOMTE_적용==========
# model.score: 0.8530612244897959
# acc_score : 0.8530612244897959
# f1_score(macro) 0.7362148163251554

'''
k_neighbors=145
model.score: 0.8765306122448979
acc_score : 0.8765306122448979
f1_score(macro) 0.6652651082750519
==========SOMTE_적용==========
model.score: 0.7959183673469388
acc_score : 0.7959183673469388
f1_score(macro) 0.6507369121221417

k_neighbors=100
model.score: 0.8765306122448979
acc_score : 0.8765306122448979
f1_score(macro) 0.6652651082750519
==========SOMTE_적용==========
model.score: 0.8051020408163265
acc_score : 0.8051020408163265
f1_score(macro) 0.6615260130652385

k_neighbors=50
model.score: 0.8765306122448979
acc_score : 0.8765306122448979
f1_score(macro) 0.6652651082750519
==========SOMTE_적용==========
model.score: 0.8173469387755102
acc_score : 0.8173469387755102
f1_score(macro) 0.6786564700109369

k_neighbors=30
model.score: 0.8765306122448979
acc_score : 0.8765306122448979
f1_score(macro) 0.6652651082750519
==========SOMTE_적용==========
model.score: 0.8214285714285714
acc_score : 0.8214285714285714
f1_score(macro) 0.6891811585934823

k_neighbors=1
model.score: 0.8765306122448979
acc_score : 0.8765306122448979
f1_score(macro) 0.6652651082750519
==========SOMTE_적용==========
model.score: 0.8418367346938775
acc_score : 0.8418367346938775
f1_score(macro) 0.6903335323236405

'''