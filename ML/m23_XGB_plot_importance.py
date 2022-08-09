import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target
print(datasets.feature_names)

x=pd.DataFrame(x,columns=[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']])

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
# model3=GradientBoostingRegressor()
model4=XGBRegressor()

# 3. 훈련
# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
# model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

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

# def plot_feature_importances(model):
#     n_features=datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1,n_features)
#     plt.title(model3)

# plot_feature_importances(model3)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model4)
plt.show()

# 판다스의 데이터 프레임으로 변경하면 컬럼명으로 볼 수 있다.
