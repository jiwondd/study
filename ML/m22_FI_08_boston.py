import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import r2_score

datasets=load_boston()

#1. 데이터
x=datasets.data
y=datasets.target

x = np.delete(x,[1,8,11],axis=1)
# print(x.shape) #(506, 13)->(506, 10)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.7,shuffle=True, random_state=777)

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

# DecisionTreeRegressor() 의 model.score: 0.8136811304794908
# DecisionTreeRegressor() 의 r2_score : 0.8136811304794908
# DecisionTreeRegressor() : [3.48669336e-02 1.52841082e-03 4.09303087e-03 3.87993209e-04
#  4.17165723e-02 5.83925459e-01 9.06939405e-03 8.07985194e-02
#  5.54191869e-03 1.38163495e-02 5.35683476e-03 1.66627300e-02
#  2.02235854e-01]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.8551143250921796
# RandomForestRegressor() 의 r2_score : 0.8551143250921796
# RandomForestRegressor() : [0.02189021 0.00088951 0.00721699 0.00078615 0.03477263 0.42984324
#  0.01313283 0.05113636 0.0060229  0.01411483 0.01024109 0.01290263
#  0.39705063]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.8704303773311544
# GradientBoostingRegressor() 의 r2_score : 0.8704303773311544
# GradientBoostingRegressor() : [0.00937382 0.         0.00159676 0.00062585 0.04405807 0.35473441
#  0.01636848 0.06799904 0.00167478 0.01290921 0.03045175 0.00953704
#  0.4506708 ]
# *************************************************
# XGBRegressor의 model.score: 0.8555128443041007
# XGBRegressor의 r2_score : 0.8555128443041007
# XGBRegressor: [0.01393674 0.00063034 0.01524248 0.01014077 0.06837803 0.23163718
#  0.01681772 0.05540737 0.0084176  0.03242847 0.039013   0.00813061
#  0.4998197 ]

# [1,8,11] 컬럼제거 / 전반적으로 성능 하락함....
# DecisionTreeRegressor() 의 model.score: 0.7134972140050884
# DecisionTreeRegressor() 의 r2_score : 0.7134972140050884
# DecisionTreeRegressor() : [3.61679469e-02 6.77815299e-03 1.90467714e-04 3.92085649e-02
#  5.87751771e-01 1.58527420e-02 8.40393252e-02 1.15671764e-02
#  6.99248703e-03 2.11451366e-01]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.840475916406877
# RandomForestRegressor() 의 r2_score : 0.840475916406877
# RandomForestRegressor() : [0.02481268 0.00691886 0.00089957 0.03109643 0.398276   0.01337035
#  0.05370058 0.01517683 0.01145611 0.44429258]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.859276430947178
# GradientBoostingRegressor() 의 r2_score : 0.859276430947178
# GradientBoostingRegressor() : [0.01247228 0.00098199 0.0010303  0.04649852 0.35491196 0.01680178
#  0.06784071 0.01485931 0.02848252 0.45612062]
# *************************************************
# XGBRegressor 의 model.score: 0.8572554008312276
# XGBRegressor 의 r2_score : 0.8572554008312276
# XGBRegressor : [0.01316839 0.01431282 0.01123578 0.0663735 
#  0.24752884 0.01461499
#  0.05390718 0.0456361  0.04642864 0.4867938 ]