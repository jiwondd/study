from unittest import skip
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
# print(sk.__version__) #0.24.2
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets=load_boston()
x=datasets.data
y=datasets.target
# print(x.shape,y.shape) (506, 13) (506,)
pca=PCA(n_components=12)
x=pca.fit_transform(x)
# print(x.shape) #(506, 2)

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                                  train_size=0.8,random_state=123,shuffle=True)

# 2. 모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model=RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train) #eval_metric='error' 스포입니다

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('결과:',result)

# 결과: 0.7912032477461212 <-befor pca
# 결과: 0.3402247421788629 <-n_components=2
# 결과: 0.37541760103245514 <- n_components=4
# 결과: 0.7204040543258998 <- n_components=7
# 결과: 0.7885739720891265 <- n_components=10
# 결과: 0.7634099128630358 <- n_components=12


