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
# print(data_set.describe())
# print(data_set.info())

data_set=data_set.to_numpy()

x=data_set[:,:11]
y=data_set[:,11]

# def outliers(data_out):
#     quartile_1, q2, quartile_3=np.percentile(data_out,
#                                        [25,50,75])
#     print("1사분위:",quartile_1)
#     print("q2:",q2)
#     print("3사분위:",quartile_3)
#     iqr=quartile_3-quartile_1
#     print("ipr : " ,iqr) 
#     lower_bound=quartile_1-(iqr*1.5) 
#     upper_bound=quartile_3+(iqr*1.5) 
#     return np.where((data_out>upper_bound)|
#                     (data_out<lower_bound))

# outliers_loc=outliers(data_set)
# print("이상치의 위치 : ",outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(data_set)
# plt.show()


# 이상치 제거해보기


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

# model.score: 0.7030612244897959
# acc_score : 0.7030612244897959
# f1_score(macro) 0.4216568183429731
# f1_score(micro) 0.7030612244897959

