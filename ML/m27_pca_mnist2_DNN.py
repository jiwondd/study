# [실습]
# 아까 4가지 모델 맹글기

import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

(x_train,y_train),(x_test,y_test)=mnist.load_data()

# print(x_train.shape,x_test.shape) #(60000, 28, 28) (10000, 28, 28)
x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)
# print(x_train.shape,x_test.shape) (60000, 784) (10000, 784)

pca=PCA(n_components=713)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
# print(x_train.shape,x_test.shape) 
# pca_EVR=pca.explained_variance_ratio_
# cumsum=np.cumsum(pca_EVR)

# print(np.argmax(cumsum>=0.95)+1) #154
# print(np.argmax(cumsum>=0.99)+1) #331
# print(np.argmax(cumsum>=0.999+1)) #486
# print(np.argmax(cumsum>=1.0)+1) #713

x_train, x_test, y_train, y_test=train_test_split(x_train,y_train,train_size=0.8,stratify=y_train,
                                                  random_state=123,shuffle=True)

# print(x_train.shape) (48000, 154)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
model=XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train) 
end=time.time()

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('결과:',result)
print('걸린시간 :',np.round(end-start,2))


'''
1. dnn
accuracy :  0.9699166417121887
걸린시간: 220.51

2. cnn
accuracy :  0.9788333177566528
걸린시간: 199.33

3. PCA 0.95
결과: 0.9608333333333333
걸린시간 : 211.31

4. PCA 0.99
결과: 0.957
걸린시간 : 434.9

5. PCA 0.999
결과: 0.9569166666666666
걸린시간 : 648.08

6. PCA 1.0
결과: 0.958
걸린시간 : 948.37

결과: 0.9583333333333334
걸린시간 : 22.2 <-gpu개쩐다리

'''