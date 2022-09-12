from unittest import skip
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
# print(sk.__version__) #0.24.2
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target
# print(x.shape,y.shape) (569, 30) (569,)
pca=PCA(n_components=15)
x=pca.fit_transform(x)
# print(x.shape) #(506, 2)
pca_EVR=pca.explained_variance_ratio_
print(pca_EVR) #새로생긴 피쳐의 중요도
# [9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
#  8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
#  3.44135279e-07 1.86018721e-07 6.99473205e-08 1.65908880e-08
#  6.99641650e-09 4.78318306e-09 2.93549214e-09 1.41684927e-09
#  8.29577731e-10 5.20405883e-10 4.08463983e-10 3.63313378e-10
#  1.72849737e-10 1.27487508e-10 7.72682973e-11 6.28357718e-11
#  3.57302295e-11 2.76396041e-11 8.14452259e-12 6.30211541e-12
#  4.43666945e-12 1.55344680e-12]
print(sum(pca_EVR)) #0.9999999999999998

# 일일히 해보면서 값을 찾지말고 그래프를 그려서 한눈에 보자

cumsum=np.cumsum(pca_EVR) #누적합계
print(cumsum)
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


'''
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

'''
