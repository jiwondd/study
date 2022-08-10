import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold
# grid-> 모눈종이처럼 촘촘하게 넣을게! CV=cross_val
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
datasets=load_iris()
x=datasets['data']
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234)

n_splits=5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

parameters=[
    {"C":[1,10,100,1000],"kernel":["linear"],"degree":[3,4,5]}, #12번
    {"C":[1,10,100],"kernel":["rbf"],"gamma":[0.001,0.0001]},   #6번
    {"C":[1,10,100,1000],"kernel":["sigmoid"],                  #24번
     "gamma":[0.01,0.001,0.0001],"degree":[3,4]}                #총 42번
]

#2. 모델구성

model=SVC(C=1, kernel='linear',degree=3)
# model=GridSearchCV(SVC(),parameters,cv=kfold, verbose=1,
#                    refit=True, n_jobs=-1)
# n-jobs = -1 cpu 다쓸거임

#3. 컴파일, 훈련
model.fit(x_train,y_train)

# Fitting 5 folds for each of 42 candidates, totalling 210 fits

# # print("최적의 매개변수: ",model.best_estimator_)
# # # 최적의 매개변수: SVC(C=1, kernel='linear')
# # print("최적의 파라미터: ",model.best_params_)
# # # 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# print("best_score: ",model.best_score_)
# # best_score:  0.975 얘는 트레인셋의 베스트스코어
print("model.score:",model.score(x_test,y_test))
# model.score: 0.9333333333333333 

#4. 평가, 예측
y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)
# y_pred_best=model.best_estimator_.predict(x_test)
# print("최적 튠 acc : ",accuracy_score(y_test,y_pred_best))

# acc score : 0.9333333333333333
# 최적 튠 acc :  0.9333333333333333

# model.score: 1.0
# acc score : 1.0 
