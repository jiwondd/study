import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_iris, load_wine
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder

# datasets=load_iris() (150, 4)->(150, 2) / [0.9912126 1.       ]
# datasets=load_breast_cancer() (569, 30)->(569, 1) / [1.]
# datasets=load_wine() (178, 13)->(178, 2) / [0.68747889 1.        ]
datasets=fetch_covtype() #(581012, 54)->(581012, 6) / [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# datasets=load_digits() #(1797, 64)->(1797, 9) / [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
#  0.94984789 0.9791736  1.        ]

x=datasets.data
y=datasets.target
print(x.shape)
print(np.unique(y, return_counts=True))
print('----------------LDA_적용후----------------')

lda=LinearDiscriminantAnalysis() #디폴트 = 라벨의 갯수 -1
lda.fit(x,y)
x=lda.transform(x)
print(x.shape)

lda_EVR=lda.explained_variance_ratio_
cumsum=np.cumsum(lda_EVR)
print(cumsum)

