from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

x = np.delete(x,[3,4],axis=1)
# print(x.shape) #(20640, 8)->(20640, 6)


#1. 데이터
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=42)

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

# DecisionTreeRegressor() 의 model.score: 0.6058512874479592
# DecisionTreeRegressor() 의 r2_score : 0.6058512874479592
# DecisionTreeRegressor() : [0.52753826 0.05219827 0.05378594 0.02798331 0.03031428 0.13087354
#  0.09277235 0.08453405]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.80690501095807
# RandomForestRegressor() 의 r2_score : 0.80690501095807
# RandomForestRegressor() : [0.52579991 0.05452899 0.04418887 0.02968456 0.03053088 0.13804192
#  0.08877851 0.08844637]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.7756446042829697
# GradientBoostingRegressor() 의 r2_score : 0.7756446042829697
# GradientBoostingRegressor() : [0.60423685 0.03413067 0.0241255  0.0049365  0.00135411 0.12283212
#  0.09853146 0.10985277]
# *************************************************
# XGBRegressor의 model.score: 0.828616180679985
# XGBRegressor의 r2_score : 0.828616180679985
# XGBRegressor: [0.4463449  0.07563571 0.04291126 0.02547988 0.02304496 0.15542509
#  0.10923825 0.12191991]

# [3,4] 컬럼제거 / 전반적으로 비슷했다.
# DecisionTreeRegressor() 의 model.score: 0.6156949157570544
# DecisionTreeRegressor() 의 r2_score : 0.6156949157570544
# DecisionTreeRegressor() : [0.53362354 0.05876462 0.06515158 0.14058894 0.1053739  0.09649741]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.8099495991369874
# RandomForestRegressor() 의 r2_score : 0.8099495991369874
# RandomForestRegressor() : [0.5327039  0.05989128 0.05792739 0.1474516  0.10137073 0.1006551 ]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.774127201598537
# GradientBoostingRegressor() 의 r2_score : 0.774127201598537
# GradientBoostingRegressor() : [0.60505384 0.03455947 0.02549026 0.12487605 0.09847482 0.11154555]
# *************************************************
# XGBRegressor의 model.score: 0.8347812916100907
# XGBRegressor의 r2_score : 0.8347812916100907
# XGBRegressor: [0.486666   0.079212   0.04827353 0.14549156 0.11509748 0.12525946]