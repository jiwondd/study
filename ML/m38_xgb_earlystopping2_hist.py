import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target
print(x.shape,y.shape) #569, 30) (569,)

x_train, x_test, y_train, y_test=train_test_split(x,y,
                                                  shuffle=True,random_state=123,train_size=0.8,stratify=y)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=123)

# 2. 모델
model=XGBClassifier(random_state=123,
                    n_estimator=100,
                    learnig_rate=0.3,
                    max_depth=6,
                    gamma=0)

# 3. 훈련
model.fit(x_train,y_train,early_stopping_rounds=10,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          eval_metric='error'
          )

# 4. 평가
result=model.score(x_test,y_test)
print('model.score:',result) 
y_predict=model.predict(x_test)
acc=accuracy_score(y_test, y_predict)
print('진짜 최종 test 점수 : ' , acc)
print('---------------------------------------------------')
hist=model.evals_result()
print(hist)

import matplotlib.pyplot as plt
# 서브플롯이용해서 두개의 그래프로 그리기 
plt.subplot(2, 1, 1)  
plt.plot(hist['validation_0']['error'],marker='.',c='red',label='loss') 
plt.grid() #모눈종이ㄱ
plt.ylabel('loss')
plt.xlabel('number of tree')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)  
plt.plot(hist['validation_1']['error'],label="val_1",marker='.',c='blue') 
plt.grid() #모눈종이ㄱ
plt.ylabel('loss')
plt.xlabel('number of tree')
plt.legend(loc='upper right')
plt.show()

# 한 그래프안에 그리기 
plt.plot(hist['validation_0']['error'],marker='.',c='red',label='loss') 
plt.plot(hist['validation_1']['error'],marker='.',c='blue',label="val_1") 
plt.grid() #모눈종이ㄱ
plt.ylabel('loss')
plt.xlabel('number of tree')
plt.legend(loc='upper right')
plt.show()



# loss:0.03059
# model.score: 0.9912280701754386

# model.score: 0.9912280701754386

# loss:0.03059
# model.score: 0.9912280701754386