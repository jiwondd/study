# 피쳐하나 삭제해서 비교해보기

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target
# print(x.shape) #(442, 10)
x = np.delete(x,1,axis=1)
# print(x.shape) (442, 9)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8, shuffle=True, random_state=1004)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model1=DecisionTreeRegressor()
model2=RandomForestRegressor()
model3=GradientBoostingRegressor()
model4=XGBRegressor()

# 3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4.평가, 예측
result1=model1.score(x_test,y_test)
print(model1,'의 model.score:',result1)
y_predict1=model1.predict(x_test)
r21=r2_score(y_test,y_predict1)
print(model1,'의 r2_score :',r21)
print(model1,':',model1.feature_importances_)
print('*************************************************')
result2=model2.score(x_test,y_test)
print(model2,'의 model.score:',result2)
y_predict2=model2.predict(x_test)
r22=r2_score(y_test,y_predict2)
print(model2,'의 r2_score :',r22)
print(model2,':',model2.feature_importances_)
print('*************************************************')
result3=model3.score(x_test,y_test)
print(model3,'의 model.score:',result3)
y_predict3=model3.predict(x_test)
r23=r2_score(y_test,y_predict3)
print(model3,'의 r2_score :',r23)
print(model3,':',model3.feature_importances_)
print('*************************************************')
result4=model4.score(x_test,y_test)
print(model4,'의 model.score:',result4)
y_predict4=model4.predict(x_test)
r24=r2_score(y_test,y_predict4)
print(model4,'의 r2_score :',r24)
print(model4,':',model4.feature_importances_)
print('*************************************************')

# DecisionTreeRegressor() 의 model.score: 0.09098083952038294
# DecisionTreeRegressor() 의 r2_score : 0.09098083952038294
# DecisionTreeRegressor() : [0.05346365 0.02337012 0.37153698 0.11428196 0.05372779 0.07915234
#  0.06438812 0.00734281 0.15211061 0.08062562]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.5026090606564253
# RandomForestRegressor() 의 r2_score : 0.5026090606564253
# RandomForestRegressor() : [0.06350872 0.01462855 0.28000276 0.12999916 0.04606998 0.05741783
#  0.05511284 0.02091841 0.2547616  0.07758015]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.48576464535317765
# GradientBoostingRegressor() 의 r2_score : 0.48576464535317765
# GradientBoostingRegressor() : [0.05204386 0.02383814 0.30152137 0.13205653 0.02585844 0.05838597
#  0.03670594 0.01845323 0.29157857 0.05955794]
# *************************************************
# XGBRegressor의 model.score: 0.38476764985992584
# XGBRegressor의 r2_score : 0.38476764985992584
# XGBRegressor: [0.0300432  0.06629263 0.19166744 0.09721985 0.04002731 0.05920865
#  0.03668924 0.12904641 0.29130363 0.05850172]


# DecisionTreeRegressor() 의 model.score: 0.05286708043194943
# DecisionTreeRegressor() 의 r2_score : 0.05286708043194943
# DecisionTreeRegressor() : [0.05779164 0.37955764 0.11221414 0.06987747 0.07544686 0.0579512
#  0.00493572 0.15435282 0.08787252]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.4882046753442122
# RandomForestRegressor() 의 r2_score : 0.4882046753442122
# RandomForestRegressor() : [0.05906604 0.27150903 0.13950724 0.04977907 0.05768899 0.05907085
#  0.02053978 0.25940416 0.08343485]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.4812950372400959
# GradientBoostingRegressor() 의 r2_score : 0.4812950372400959
# GradientBoostingRegressor() : [0.04707582 0.30498571 0.13140844 0.03077887 0.06138468 0.03075722
#  0.02440665 0.30443759 0.06476503]
# *************************************************
# XGBRegressor의 model.score: 0.4454704009902666
# XGBRegressor의 r2_score : 0.4454704009902666
# XGBRegressor: [0.02858853 0.19733451 0.11821878 0.03948803 0.06846352 0.06047699
#  0.10674872 0.29530895 0.08537202]


# 덜 중요하다고 해서 빼도 성능차이는 뭐 별루...?
