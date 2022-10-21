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
          eval_set=[(x_test,y_test)],
          eval_metric='error'
          )

# 4. 평가
result=model.score(x_test,y_test)
print('model.score:',result) 
y_predict=model.predict(x_test)
acc=accuracy_score(y_test, y_predict)
print('진짜 최종 test 점수 : ' , acc)

# import pickle
path='d:/study_data/_save/_xg/'
# pickle.dump(model, open(path+'m39_pickle1_save.data','wb'))
#                       / 여기부터 여기까지 파일명  / write+binary

# import joblib
# path='d:/study_data/_save/_xg/'
# joblib.dump(model,path+'m40_joblib_save.dat')

model.save_model(path+'m41_xgb1_savemodel.dat')


# loss:0.03059
# model.score: 0.9912280701754386

# model.score: 0.9912280701754386

# loss:0.03059
# model.score: 0.9912280701754386