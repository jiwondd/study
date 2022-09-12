import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_iris, load_wine
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
# print('xg버전 : ',xg.__version__) xg버전 :  1.6.1

# 1. 데이터
# datasets=fetch_covtype()

datasets=load_iris()
x=datasets.data
y=datasets.target
# print(x.shape) (150, 4)
# print(np.unique(y, return_counts=True)) (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

# le=LabelEncoder()
# y=le.fit_transform(y)

# pca=PCA(n_components=20)
# x=pca.fit_transform(x)
# lda=LinearDiscriminantAnalysis()
# lda.fit(x,y)
# x=lda.transform(x)

# print(x.shape) #(506, 2)
# pca_EVR=pca.explained_variance_ratio_
# cumsum=np.cumsum(pca_EVR)
# print(cumsum)

lda=LinearDiscriminantAnalysis(n_components=2) 
lda.fit(x,y)
x=lda.transform(x)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

# 2. 모델구성
from xgboost import XGBClassifier, XGBRegressor
model=XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
results=model.score(x_test,y_test)
print('결과:',results)
print('걸린시간:',np.round(end-start,2))

# xgboost with gpu
# 결과: 0.8710446374017883
# 걸린시간: 5.98

# xgboost with gpu / n_components=10
# 결과: 0.8394791872843214
# 걸린시간: 4.03

# xgboost with gpu / n_components=20
# 결과: 0.8867327005326885
# 걸린시간: 4.76

# LDA
# 결과: 0.9666666666666667
# 걸린시간: 0.7



