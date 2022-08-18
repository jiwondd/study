import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets=load_iris()
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

print(model1,':',model1.feature_importances_)
print(model2,':',model2.feature_importances_)
print(model3,':',model3.feature_importances_)
print(model4,':',model4.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importances_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names) #그래프에 눈금표시
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features) #y축에 limit 지정하기
    

plt.subplot(2,2,1)
plot_feature_importances_datasets(model1)
plt.title('DecisionTreeRegressor')
plt.subplot(2,2,2)
plot_feature_importances_datasets(model2)
plt.title('RandomForestRegressor')
plt.subplot(2,2,3)
plot_feature_importances_datasets(model3)
plt.title('GradientBoostingRegressor')
plt.subplot(2,2,4)
plot_feature_importances_datasets(model4)
plt.title('XGBRegressor')
plt.show()
