import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#1. 데이터
datasets=load_digits()
x = datasets.data 
y = datasets.target

# print(np.unique(y, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

lda=LinearDiscriminantAnalysis(n_components=9) 
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

# 결과: 0.9611111111111111
# 걸린시간: 1.34