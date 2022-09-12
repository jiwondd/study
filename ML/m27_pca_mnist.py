import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist

(x_train, _),(x_test,_)=mnist.load_data() #비지도 학습
# y_train안받아 버려 / y_test 안받아 버려 그러면?!
# 라벨이 없지? 그럼? 비지도 학습이겠쥬?

# print(x_train.shape,x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x=np.append(x_train,x_test,axis=0)
# print(x.shape) #(70000, 28, 28)

###################################################################
# [실습] pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 / 0.99 / 0.999 / 1.0 / 
# 힌트 np.argmax
###################################################################

x=x.reshape(70000,28*28)
# x=x.reshape(x.shape[0],x.shape[1]*x.shape[2]) 위에랑 똑같지?
# print(x.shape) #(70000, 784)

pca=PCA(n_components=784)
x=pca.fit_transform(x)
# print(x.shape) #(506, 2)
pca_EVR=pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR)) #1.000000000000002

cumsum=np.cumsum(pca_EVR)
# print(cumsum)

# np.whrer어캐하는지 나중에 찾아봅시다.
# test=np.where(cumsum>=0.95,'pass','del')
# print(test)
# print(test.value_counts())

# print(np.argwhere(cumsum>=0.95)[0]+1) #[154]
print(np.argmax(cumsum>=0.95)+1) #154
print(np.argmax(cumsum>=0.99)+1) #331
print(np.argmax(cumsum>=0.999+1)) #486
print(np.argmax(cumsum>=1.0)+1) #713

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

'''
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
'''