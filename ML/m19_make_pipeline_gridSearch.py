import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets=load_iris()
x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=1004)

# scaler=MinMaxScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
parameters=[
    {'randomForestClassifier__n_estimators':[100,200],'randomForestClassifier__max_depth':[6,8,10,23]},
    {'randomForestClassifier__min_samples_leaf':[3,5,7,10],'randomForestClassifier__min_samples_split':[2,3,5,10],
     'randomForestClassifier__n_jobs':[-1,2,4]}
]

n_splits=5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1004)


# 2. 모델구성
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline #소문자니까 함수같아 보이지?
from sklearn.model_selection import GridSearchCV

pipe=make_pipeline(MinMaxScaler(),RandomForestClassifier())
# pipe=Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=1)
model=GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

# 3. 훈련
start=time.time()
model.fit(x_train,y_train)
end=time.time()
# 파이프라인으로 fit하면 fit_transform 으로 돌아가면서 스케일링이 같이 적용된다.

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

# print('--------------------------------------')
# y_predict=model.predict(x_test)
# acc=accuracy_score(y_test, y_predict)
# print('acc_score : ',acc)



# =========================================
# RandomForestClassifier 결과:  0.9666666666666667
# RandomForestClassifier_acc score : 0.9666666666666667

# GridSearchCV
# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# best_score:  0.9666666666666668
# model.score: 1.0
# 걸린시간: 5.93

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# best_score:  0.9583333333333334
# model.score: 1.0
# 걸린시간: 2.43

# HalvingGridSearchCV
# best_score:  0.9666666666666668
# model.score: 1.0
# 걸린시간: 8.13

# HalvingRandomSearchCV
# best_score:  0.9333333333333332
# 걸린시간: 2.24

# model.score: 1.0 <- pipeline

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# model.score: 1.0
