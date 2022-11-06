import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from icecream import ic


x=np.arange(8).reshape(4,2)

# print(x) 
# print(x.shape) 

pf=PolynomialFeatures(degree=2)
pf2=PolynomialFeatures(degree=3)
# 보통 디그리2까지 한다. 3은 너무 늘어나버려서 별루임
x_pf=pf.fit_transform(x)
x_pf2=pf2.fit_transform(x)
print('[[0 1] [2 3] [4 5] [6 7]]')
print('pf=PolynomialFeatures(degree=2)')
print(x_pf)

print('pf=PolynomialFeatures(degree=3)')
print(x_pf2)

# print(x_pf.shape) (4, 6)

#####################################

x=np.arange(12).reshape(4,3)

# print(x) [0 1] [2 3] [4 5] [6 7]
# print(x.shape) (4, 2)

pf=PolynomialFeatures(degree=2)
x_pf2=pf.fit_transform(x)

print(x_pf2)
# print(x_pf.shape) (4, 6)

#####################################
# 파이프라인에 pca/폴리/kmeans같은 애들도 붙일 수 있다.
