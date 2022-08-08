from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.model_selection import KFold, cross_val_score

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

#1. 데이터
# x_train,x_test,y_train,y_test=train_test_split(x,y,
#         train_size=0.8,shuffle=True, random_state=42)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)



#2. 모델구성
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 분류임 어그로...(논리적인회귀=분류)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
model=RandomForestRegressor()

# 3.4. 컴파일 훈련 평가 예측
scores=cross_val_score(model,x,y,cv=kfold)
# scores=cross_val_score(model,x,y,cv=5) 위에랑 똑같아요~
print('R2:',scores,'\n cross_val_score:', round(np.mean(scores),4))

# R2: [0.8235209  0.81680645 0.79475286 0.80505362 0.80700664] 
#  cross_val_score: 0.8094