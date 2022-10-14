import numpy as np
import pandas as pd

data=pd.DataFrame([[2,np.nan,6,8,10],
                   [2,4,np.nan,8,np.nan,],
                   [2,4,6,8,10],
                   [np.nan,4,np.nan,8,np.nan,]])

# print(data.shape) (4, 5)
data=data.transpose()
data.columns=['x1','x2','x3','x4']
# print(data.shape) (5, 4)

# 결측치확인
print('==========================')
print('print(data.isnull())')
print(data.isnull())
print('==========================')
print('print(data.isnull().sum())')
print(data.isnull().sum())
print('==========================')
print('print(data.info())')
print(data.info())
print('==========================')
print('기존의_data')
print(data)

# # 결측치 처리
# # 1. 결측치 삭제
# print(data.dropna()) #디폴트는 행을 지우는것
# print(data.dropna(axis=1))

# # 2-1 특정값 채우기(평균) 
# print('================평균값_채우기==================')
# means=data.mean() #전체 평균이 아니라 컬럼별 평균이니까 데이터가 겁나 틀어져버림
# print('평균 : ' , means)
# data2=data.fillna(means)
# print(data2)

# # 2-2 특정값 채우기(중간) 
# print('================중위값_채우기==================')
# median=data.median()
# print('중위 : ' , median)
# data3=data.fillna(median)
# print(data3)

# # 2-3 특정값 채우기 (앞,뒤)
# print('================앞/뒤값_채우기==================')
# data4=data.fillna(method='ffill') #첫번째가 결측치면 안들어가짐
# print(data4)
# data5=data.fillna(method='bfill') #마지막이 결측치면 안들어가짐
# print(data5)

# # 2-4 특정값 채우기 (임의값)
# print('================임의값_채우기==================')
# data6=data.fillna(7777) # 앞에랑 똑같애 data6=data.fillna(value=7777)
# print(data6)

# 특정칼럼만 특정값으로 채우기
# print('================특정칼럼/특정값_채우기==================')
# means=data['x1'].mean()
# data['x1']=data['x1'].fillna(means)
# med=data['x2'].median()
# data['x2']=data['x2'].fillna(med)
# data['x4']=data['x4'].fillna(1004)
# print(data)