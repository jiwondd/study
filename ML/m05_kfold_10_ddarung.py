import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.model_selection import KFold, cross_val_score

#.1 데이터
path='./_data/ddarung/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'submission.csv',index_col=0)

test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
train_set=train_set.dropna()
test_set=test_set.fillna(0)
x=train_set.drop(['count'],axis=1)
y=train_set['count']

# x_train,x_test,y_train,y_test=train_test_split(x,y,
#         train_size=0.99,shuffle=True, random_state=750)

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x_train=scaler.transform(x)


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

# R2: [0.76017633 0.73013795 0.80730681 0.78556262 0.79103477] 
#  cross_val_score: 0.7748