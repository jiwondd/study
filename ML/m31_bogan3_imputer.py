import numpy as np
import pandas as pd

data=pd.DataFrame([[2,np.nan,6,8,10],
                   [2,4,np.nan,8,np.nan,],
                   [2,4,6,8,10],
                   [np.nan,4,np.nan,8,np.nan,]])

# print(data.shape) (4, 5)
data=data.transpose()
data.columns=['x1','x2','x3','x4']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# imputer=SimpleImputer() #평균값으로 채워진다.
# imputer=SimpleImputer(strategy='mean') #디폴트
# imputer2=SimpleImputer(strategy='median') #중위값으로 채워진다.
# imputer=SimpleImputer(strategy='most_frequent') #자주 나왔던 값으로 채워진다.
# imputer=SimpleImputer(strategy='constant') #상수로 채운다 (디폴트 0)
# imputer2=SimpleImputer(strategy='constant',fill_value=1004) #특정값 채우기
imputer=KNNImputer() #평균값으로 채워진다.
# imputer2=IterativeImputer() 
# # [[ 2.          2.          2.          2.0000005 ] 
# #  [ 4.00000099  4.          4.          4.        ] 
# #  [ 6.          5.99999928  6.          5.9999996 ] 
# #  [ 8.          8.          8.          8.        ] 
# #  [10.          9.99999872 10.          9.99999874]]

print('****KNNImputer()*****')
imputer.fit(data)
data2=imputer.transform(data)
print(data2)


# print('****IterativeImputer()****')
# imputer2.fit(data)
# data3=imputer2.transform(data)
# print(data3)