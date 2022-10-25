import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from cProfile import label
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
#.1 데이터
path='d:/study_data/_data/'
data_set=pd.read_csv(path+'winequality-white.csv',index_col=None, sep=';')
# print(data_set.shape) (4898, 11)                                ㄴ기준으로 컬럼을 나눠줘

print(data_set.describe())
print(data_set.info())

# data_set2=data_set.to_numpy()
# data_set=data_set.values
# print(type(data_set))
# 이러면 인덱스랑 컬럼이 사라지고 넘파이 어레이 형태로 저장된다

#.1 데이터
# x=data_set2[:,:11]
# y=data_set2[:,11] #np일때 x,y나누기

y=data_set['quality']
x=data_set.drop(['quality'],axis=1) #df일때 x,y나누기

# print(x.shape,y.shape) (4898, 11) (4898,)
# print(np.unique(y,return_counts=True)) np형태일때
# print(data_set['quality'].value_counts()) df형태일때 

# le=LabelEncoder()
# y=le.fit_transform(y)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8,stratify=y,
                                                  random_state=123,shuffle=True)

# 2. 모델구성
model = RandomForestClassifier(random_state=123)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred=model.predict(x_test)
result=model.score(x_test,y_test)
print('model.score:',result) 
print('acc_score :' ,accuracy_score(y_test,y_pred))
print('f1_score(macro)',f1_score(y_test,y_pred, average='macro'))
print('f1_score(micro)',f1_score(y_test,y_pred, average='micro'))


# (랜덤스테이트 42, 1004, 777, 1234 다 별로)
# model.score: 0.47551020408163264 <-xg
# model.score: 0.6908163265306122 <-rf

# model.score: 0.7204081632653061 ...이게모죠?? 왜 이게 더...?

# model.score: 0.7204081632653061
# acc_score : 0.7204081632653061
# f1_score(macro) 0.42939251311510584
# f1_score(micro) 0.7204081632653062
