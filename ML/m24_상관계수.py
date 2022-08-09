import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. 데이터
datasets= load_iris()
# print(datasets.feature_names)
x=datasets['data']
y=datasets['target']

df=pd.DataFrame(x,columns=[['sepal length', 'sepal width', 'petal length', 'petal width']])

df['target(y)']=y
# print(df) [150 rows x 5 columns]
# 스킷런의 데이터를 판다스 데이터 프레임으로 다 바꿔줌

print('===================상관계수 히트 맵===================')
print(df.corr()) #상관관계를 볼 수 있다

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True, #정사각형으로 그려줘
            annot=True,  #각 cell의 값 표기 유무
            cbar=True    #컬러바의 유무
            )
              

plt.show()