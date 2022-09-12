import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
tf.random.set_seed(123)

#1. 데이터
datasets=load_breast_cancer()
x = datasets.data 
y = datasets.target

pca=PCA(n_components=25)
x=pca.fit_transform(x)
# print(x.shape) #(569, 30)

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                                  train_size=0.8,random_state=123,shuffle=True)

# 2. 모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model=RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train) 

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('결과:',result)

# 결과: 0.9113976612094887
# 결과: 0.9093332442365519 / n_components=4
# 결과: 0.9025762779819578 / n_components=5
# 결과: 0.8959602405613097 / n_components=10
# 결과: 0.8845754761109255 / n_components=15
# 결과: 0.8811436685599732 / n_components=20
# 결과: 0.8832537921817574 / n_components=25