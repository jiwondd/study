#  1 357 / 0 212 
# 라벨 0을 112개 삭제해서 재구성
# 그런다음 smote넣어서 맹그러서 넣은거 안넣은거 비교하기

# smote 넣은거 안넣은거 비교하기
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import time

# 1. 데이터
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target
# print(x.shape,y.shape) (569, 30) (569,)
# print(np.unique(y, return_counts=True)) (array([0, 1]), array([212, 357], dtype=int64))

zero_y = np.where(y==0)
del_y = zero_y[0][100:]
print(del_y)
new_x = np.delete(x,del_y,axis=0)
new_y = np.delete(y,del_y)
print(np.unique(new_y, return_counts=True)) #(array([0, 1]), array([100, 357], dtype=int64))
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    new_x,new_y, train_size=0.75, shuffle=True, random_state=123, stratify=new_y)

# 2. 모델
model=RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score',f1_score(y_test,y_pred))
# print(pd.Series(y_train).value_counts())
print('==========SMOTE_적용==========')
smote=SMOTE(random_state=123)#k_neighbors=2)
x_train,y_train=smote.fit_resample(x_train,y_train)
# print(pd.Series(y_train).value_counts())

model=RandomForestClassifier()
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score',f1_score(y_test,y_pred))


# model.score: 0.965034965034965
# acc_score : 0.965034965034965
# f1_score(macro) 0.9626690335717641
# ==========SMOTE_적용==========
# model.score: 0.965034965034965
# acc_score : 0.965034965034965
# f1_score(macro) 0.9717514124293786

# model.score: 0.9652173913043478
# acc_score : 0.9652173913043478
# f1_score 0.9777777777777777
# ==========SMOTE_적용==========
# model.score: 0.9565217391304348
# acc_score : 0.9565217391304348
# f1_score 0.9717514124293786
