import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# pip install imblearn 
from imblearn.over_sampling import SMOTE
import sklearn as sk

path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=None, sep=';')

y=data_set['quality']
x=data_set.drop(['quality'],axis=1)
print(np.unique(y, return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(x.shape,y.shape) (4898, 11) (4898,)

# x=x[:-40] #한개의 라벨값을 임의로 확 줄여버려
# y=y[:-40]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.75, shuffle=True, random_state=123, stratify=y)

print(pd.Series(y_train).value_counts())

# 2. 모델
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score(macro)',f1_score(y_test,y_pred, average='macro'))
print(pd.Series(y_train).value_counts()) 
print('==========SOMTE_적용==========')
smote=SMOTE(random_state=123,k_neighbors=2)
# Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
x_train,y_train=smote.fit_resample(x_train,y_train)
# print(pd.Series(y_train).value_counts()) 

model=RandomForestClassifier()
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score(macro)',f1_score(y_test,y_pred, average='macro'))

# k_neighbors=3
# acc_score : 0.673469387755102
# f1_score(macro) 0.419921110053002

# k_neighbors=2
# acc_score : 0.689795918367347
# f1_score(macro) 0.44037338791128516

