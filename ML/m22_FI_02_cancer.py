import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=load_breast_cancer()
x = datasets.data 
y = datasets.target

x = np.delete(x,[0,4,5,6,10],axis=1)
# print(x.shape) (569, 30)->(569, 25)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

#2. 모델구성
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

# DecisionTreeClassifier() 의 model.score: 0.9210526315789473
# DecisionTreeClassifier() 의 acc_score : 0.9210526315789473
# DecisionTreeClassifier() : [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.02368761
#  0.00303991 0.01205064 0.00129348 0.00744468 0.         0.
#  0.         0.00797644 0.01816057 0.01116701 0.0062039  0.15200352
#  0.00900135 0.         0.01256289 0.72634406 0.00906395 0.        ]
# *************************************************
# RandomForestClassifier() 의 model.score: 0.956140350877193
# RandomForestClassifier() 의 acc_score : 0.956140350877193
# RandomForestClassifier() : [0.04817602 0.01936454 0.0397838  0.04890266 0.0083822  0.02092248
#  0.05539396 0.08195957 0.00427236 0.00291985 0.01119238 0.00190884
#  0.01984696 0.03731857 0.00393982 0.00394772 0.00487693 0.00439698
#  0.00462285 0.00379428 0.14189548 0.01883051 0.13686124 0.11001974
#  0.01522111 0.01282173 0.034025   0.08384259 0.01398797 0.00657187]
# *************************************************
# GradientBoostingClassifier() 의 model.score: 0.9298245614035088
# GradientBoostingClassifier() 의 acc_score : 0.9298245614035088
# GradientBoostingClassifier() : [9.47086346e-05 1.75386515e-02 1.20665084e-02 4.00513311e-05
#  0.00000000e+00 2.67848031e-03 5.08532958e-03 9.21055180e-03
#  1.18417102e-04 2.67031404e-04 6.31017945e-03 2.42876948e-04
#  2.48138699e-03 1.08802689e-02 5.69978380e-04 3.60799683e-03
#  1.53151523e-02 1.20992214e-04 4.61995304e-04 4.61853555e-04
#  2.35126493e-01 3.01712800e-02 5.53155790e-02 4.67694683e-02
#  4.67537341e-03 1.20713612e-03 1.14098423e-02 5.26863220e-01
#  1.51386098e-04 7.57811491e-04]
# *************************************************
# XGBClassifier의 model.score: 0.9210526315789473
# XGBClassifier의 acc_score : 0.9210526315789473
# XGBClassifier: [0.0005704  0.01552387 0.         0.01343192 0.00623005 0.01095381   192 0.00623005 0.01095381
#  0.01709487 0.01413621 0.01391366 0.0059227  0.00862221 0.
#  0.01242374 0.00753322 0.00471901 0.00201622 0.00137453 0.00063779
#  0.00071296 0.         0.46573263 0.01409463 0.1626587  0.03217962
#  0.01020905 0.         0.01284548 0.16369273 0.00276994 0.        ]

# [0,4,5,6,10] 번째 컬럼삭제 / XGBClassifier만 조금 좋아짐
# DecisionTreeClassifier() 의 model.score: 0.9122807017543859
# DecisionTreeClassifier() 의 acc_score : 0.9122807017543859
# DecisionTreeClassifier() : [0.02464445 0.         0.         0.         0.         0.
#  0.         0.00303991 0.02095803 0.         0.         0.
#  0.         0.         0.0253643  0.00457869 0.         0.
#  0.15200352 0.00744468 0.0062039  0.01256289 0.73203263 0.
#  0.01116701]
# *************************************************
# RandomForestClassifier() 의 model.score: 0.9473684210526315
# RandomForestClassifier() 의 acc_score : 0.9473684210526315
# RandomForestClassifier() : [0.02911294 0.04655192 0.08931184 0.11465685 0.00562574 0.00446389
#  0.00651354 0.01046007 0.0413726  0.00386185 0.00406443 0.0056133
#  0.00584748 0.00583329 0.00551769 0.06910759 0.01517104 0.15417621
#  0.15142188 0.01163508 0.02099177 0.06059655 0.11650994 0.01475682
#  0.00682566]
# *************************************************
# GradientBoostingClassifier() 의 model.score: 0.9298245614035088
# GradientBoostingClassifier() 의 acc_score : 0.9298245614035088
# GradientBoostingClassifier() : [1.57092388e-02 1.90274199e-03 4.01212264e-04 1.37071980e-02  
#  3.11324756e-04 3.30325751e-04 2.12716443e-04 3.03268585e-03
#  1.40151237e-02 8.70290445e-04 3.87714127e-03 8.04438536e-03
#  3.53929705e-03 7.51645407e-04 2.38345426e-03 2.33953544e-01
#  2.74315518e-02 6.83366762e-02 5.06356911e-02 6.75725154e-03
#  3.60792035e-03 1.21840760e-02 5.27324277e-01 1.43695404e-04
#  5.36535379e-04]
# *************************************************
# XGBClassifier의 model.score: 0.9385964912280702
# XGBClassifier의 acc_score : 0.9385964912280702
# XGBClassifier: [0.02459858 0.01640118 0.01071173 0.02654469 0.02786778 0.00660581
#  0.         0.01298439 0.00927425 0.0050764  0.00259729 0.00546786
#  0.00190911 0.0010815  0.00756168 0.38921112 0.01641363 0.12726666
#  0.07099875 0.02091877 0.00516569 0.01142334 0.19540435 0.00451545
#  0.        ]