from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
#                                 ㄴ이상치에 자유로운편
from sklearn.pipeline import make_pipeline
from icecream import ic
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,accuracy_score

# 1.데이터
datasets=load_diabetes()
x,y=datasets.data,datasets.target
# ic(x.shape,y.shape)
# x.shape: (506, 13), y.shape: (506,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=KFold(n_splits=5,shuffle=True,random_state=123)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model=LinearRegression()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_predict=model.predict(x_test)
result=r2_score(y_test,y_predict)
print('그냥 결과 : ', round(result,3)) #그냥 결과 :  0.767 <-리니어 /그냥 결과 :  0.915 <-랜포


##################로그변환####################
df=pd.DataFrame(datasets.data,columns=[datasets.feature_names])
print(df)

# import matplotlib.pyplot as plt
# df.plot.box()
# plt.title('boston')
# plt.xlabel('Feature')
# plt.ylabel('data')
# plt.show()

print(df.head())                # 0.767 <-기존
df['B']=np.log1p(df['B'])       # 0.771
df['CRIM']=np.log1p(df['CRIM']) # 0.76
df['ZN']=np.log1p(df['ZN'])     # 0.773
df['TAX']=np.log1p(df['TAX'])   # 0.767
print(df.head())

x_train,x_test,y_train,y_test=train_test_split(df,y,
        train_size=0.8,shuffle=True, random_state=1234)

kfold=KFold(n_splits=5,shuffle=True,random_state=123)

# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

# 2. 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model=LinearRegression()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_predict=model.predict(x_test)
result=r2_score(y_test,y_predict)
print('로그 결과 : ', round(result,3)) 