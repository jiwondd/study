import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8, shuffle=True, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

# model1=DecisionTreeRegressor()
# model2=RandomForestRegressor()
model3=GradientBoostingRegressor()
# model4=XGBRegressor()

# 3. 훈련
# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
# model4.fit(x_train,y_train)

# 4.평가, 예측
# result1=model1.score(x_test,y_test)
# print(model1,'의 model.score:',result1)
# y_predict1=model1.predict(x_test)
# r21=r2_score(y_test,y_predict1)
# print(model1,'의 r2_score :',r21)
# print(model1,':',model1.feature_importances_)
# print('*************************************************')
# result2=model2.score(x_test,y_test)
# print(model2,'의 model.score:',result2)
# y_predict2=model2.predict(x_test)
# r22=r2_score(y_test,y_predict2)
# print(model2,'의 r2_score :',r22)
# print(model2,':',model2.feature_importances_)
# print('*************************************************')
# result3=model3.score(x_test,y_test)
# print(model3,'의 model.score:',result3)
# y_predict3=model3.predict(x_test)
# r23=r2_score(y_test,y_predict3)
# print(model3,'의 r2_score :',r23)
# print(model3,':',model3.feature_importances_)
# print('*************************************************')
# result4=model4.score(x_test,y_test)
# print(model4,'의 model.score:',result4)
# y_predict4=model4.predict(x_test)
# r24=r2_score(y_test,y_predict4)
# print(model4,'의 r2_score :',r24)
# print(model4,':',model4.feature_importances_)
# print('*************************************************')

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features=datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(model3)

plot_feature_importances(model3)
plt.show()




# DecisionTreeRegressor() 의 model.score: 0.040792550518523374  
# DecisionTreeRegressor() 의 r2_score : 0.040792550518523374    
# DecisionTreeRegressor() : [0.07305304 0.01805365 0.34887664 0.07930391 0.03833596 0.09716135
#  0.04630901 0.02583264 0.15664794 0.11642587]
# *************************************************
# RandomForestRegressor() 의 model.score: 0.40748938494946096   
# RandomForestRegressor() 의 r2_score : 0.40748938494946096     
# RandomForestRegressor() : [0.05733318 0.01233235 0.34851274 0.08105526 0.04575332 0.06158784
#  0.05614628 0.03290764 0.22103493 0.08333647]
# *************************************************
# GradientBoostingRegressor() 의 model.score: 0.41531729560820485
# GradientBoostingRegressor() 의 r2_score : 0.41531729560820485 
# GradientBoostingRegressor() : [0.04599844 0.0154556  0.33622869 0.09596942 0.03079137 0.0670519
#  0.03822083 0.01419952 0.27735927 0.07872495]
# *************************************************
#  [0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191       213588 0.0620191
#  0.06551369 0.17944618 0.13779876 0.08540721]