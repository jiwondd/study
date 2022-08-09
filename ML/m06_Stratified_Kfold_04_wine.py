import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold

#1. 데이터
datasets=load_wine()
x=datasets['data']
y=datasets.target

# x_train,x_test,y_train,y_test=train_test_split(x,y,
#         train_size=0.8,shuffle=True, random_state=31)

n_splits=5
# kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)
# 다중분류에는 라벨의갯수에 비례해서 자른다.

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x_train=scaler.transform(x)


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 분류임 어그로...(논리적인회귀=분류)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier

model=LinearSVC()

# 3.4. 컴파일 훈련 평가 예측
scores=cross_val_score(model,x,y,cv=kfold)
# scores=cross_val_score(model,x,y,cv=5) 위에랑 똑같아요~
print('ACC:',scores,'\n cross_val_score:', round(np.mean(scores),4))

# ACC: [0.61111111 0.88888889 0.80555556 0.85714286 0.94285714]
#  cross_val_score: 0.8211