import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# pip install imblearn 
from imblearn.over_sampling import SMOTE
import sklearn as sk
# print(sk.__version__) 1.1.2

# 1. 데이터
datasets=load_wine()
x=datasets.data
y=datasets['target']
# print(x.shape,y.shape) (178, 13) (178,)
# print(type(x)) #<class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.Series(y).value_counts())    [  1    71 / 0    59 / 2    48  ]
# pd.dataframe은 행렬 전체를 위에꺼는 한줄만
# print(y) 

# x=x[:-40] #한개의 라벨값을 임의로 확 줄여버려
# y=y[:-40]
# print(pd.Series(y_new).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.75, shuffle=True, random_state=123, stratify=y)

# print(pd.Series(y_train).value_counts()) #[  1    53 / 0    44 / 2    6  ]

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
# print('f1_score(micro)',f1_score(y_test,y_pred, average='micro'))

# 기본결과
# acc_score : 0.9777777777777777
# f1_score(macro) 0.9797235023041475

# 자른거 (2라벨을 30개로 줄인거)
# acc_score : 0.972972972972973
# f1_score(macro) 0.9797235023041475

# 작은범위의 데이터를 더 잘 못 맞췄다
# acc_score : 0.9428571428571428
# f1_score(macro) 0.8596176821983273

print('==========SOMTE_적용==========')
smote=SMOTE(random_state=123)
x_train,y_train=smote.fit_resample(x_train,y_train)
print(np.unique(y, return_counts=True))
# print(pd.Series(y_train).value_counts()) [  0    53 / 1    53 / 2    53 ]

model=RandomForestClassifier()
model.fit(x_train,y_train)

# test데이터는(평가데이터_) 스케일 하면 안됨 건들지말자

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score(macro)',f1_score(y_test,y_pred, average='macro'))

# model.score: 0.9428571428571428
# acc_score : 0.9428571428571428
# f1_score(macro) 0.8596176821983273
# ==========SOMTE_적용==========
# model.score: 0.9714285714285714
# acc_score : 0.9714285714285714
# f1_score(macro) 0.9797235023041475