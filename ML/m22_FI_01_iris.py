# 피쳐 임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성 후
# 모델별로 돌려서 결과도출/ 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets=load_iris()
x=datasets.data
y=datasets.target
x = np.delete(x,1,axis=1)
# print(x.shape) (150, 3)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8, shuffle=True, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model1=DecisionTreeClassifier()
model2=RandomForestClassifier()
model3=GradientBoostingClassifier()
model4=XGBClassifier()

# 3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4.평가, 예측
result1=model1.score(x_test,y_test)
print(model1,'의 model.score:',result1)
y_predict1=model1.predict(x_test)
acc1=accuracy_score(y_test,y_predict1)
print(model1,'의 acc_score :',acc1)
print(model1,':',model1.feature_importances_)
print('*************************************************')
result2=model2.score(x_test,y_test)
print(model2,'의 model.score:',result2)
y_predict2=model2.predict(x_test)
acc2=accuracy_score(y_test,y_predict2)
print(model2,'의 acc_score :',acc2)
print(model2,':',model2.feature_importances_)
print('*************************************************')
result3=model3.score(x_test,y_test)
print(model3,'의 model.score:',result3)
y_predict3=model3.predict(x_test)
acc3=accuracy_score(y_test,y_predict3)
print(model3,'의 acc_score :',acc3)
print(model3,':',model3.feature_importances_)
print('*************************************************')
result4=model4.score(x_test,y_test)
print(model4,'의 model.score:',result4)
y_predict4=model4.predict(x_test)
acc4=accuracy_score(y_test,y_predict4)
print(model4,'의 acc_score :',acc4)
print(model4,':',model4.feature_importances_)
print('*************************************************')

# DecisionTreeClassifier() 의 model.score: 1.0
# DecisionTreeClassifier() 의 acc_score : 1.0
# DecisionTreeClassifier() : [0.03338202 0.         0.05989984 0.90671814]
# *************************************************
# RandomForestClassifier() 의 model.score: 1.0
# RandomForestClassifier() 의 acc_score : 1.0
# RandomForestClassifier() : [0.09323932 0.02892302 0.44699614 0.43084153]
# *************************************************
# GradientBoostingClassifier() 의 model.score: 1.0
# GradientBoostingClassifier() 의 acc_score : 1.0
# GradientBoostingClassifier() : [0.00693848 0.01179316 0.28634043 0.69492793]    
# *************************************************
# XGBClassifier의 model.score: 1.0
# XGBClassifier의 acc_score : 1.0
# XGBClassifier: [0.00912187 0.0219429  0.678874   0.29006115]

# 두번째 컬럼 삭제 후 
# DecisionTreeClassifier() 의 model.score: 1.0
# DecisionTreeClassifier() 의 acc_score : 1.0
# DecisionTreeClassifier() : [0.01669101 0.58410048 0.39920851]
# *************************************************
# RandomForestClassifier() 의 model.score: 1.0
# RandomForestClassifier() 의 acc_score : 1.0
# RandomForestClassifier() : [0.1776631  0.43067799 0.39165891]
# *************************************************
# GradientBoostingClassifier() 의 model.score: 1.0
# GradientBoostingClassifier() 의 acc_score : 1.0
# GradientBoostingClassifier() : [0.01413338 0.36958928 0.61627734]
# *************************************************
# XGBClassifier의 model.score: 1.0
# XGBClassifier의 acc_score : 1.0
# XGBClassifier: [0.01531648 0.71701634 0.26766717] 