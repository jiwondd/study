import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)

le=LabelEncoder()
y=le.fit_transform(y_train)

# print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 
# 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

lda=LinearDiscriminantAnalysis() 
lda.fit(x_train,y_train)
x_train=lda.transform(x_train)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

n_splits= 5
kfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

Parameter=[
    {"n_estimators":[100,200],"learning_rate":[0.001,0.01],"max_depth":[5,6]},
    {"n_estimators":[50,100],"learning_rate":[0.1,0.001],"max_depth":[5,6],"colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[100,150],"learning_rate":[0.1,0.5],"max_depth":[5,6],"colsample_bylevel":[0.6,0.7,0.9]}
]

# 2. 모델구성
model=GridSearchCV(XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0),Parameter,cv=kfold, verbose=2,
                   refit=True,n_jobs=-1)

# 3. 컴파일 훈련
import time
start=time.time()
model.fit(x_train,y_train,verbose=2) 
end=time.time()

# 4. 평가 예측
result=model.score(x_test,y_test)
print('결과:',result)
print('걸린시간 :',np.round(end-start,2))

