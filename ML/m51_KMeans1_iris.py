from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from icecream import ic

datasets=load_iris()
df=pd.DataFrame(datasets.data,columns=[datasets.feature_names])
# print(df) [150 rows x 4 columns]

kmeans=KMeans(n_clusters=3,random_state=1234)
#             ㄴ라벨의 갯수
kmeans.fit(df)
print('kmeans로 만들어낸 y라벨')
print(kmeans.labels_) #kmeans적용한 y라벨보기
print('기존 datasets의 y라벨')
print(datasets.target) #원래 y라벨보기

df['cluster']=kmeans.labels_
df['target']=datasets.target

acc=accuracy_score(kmeans.labels_,datasets.target)
# 실제 y값과 kmeans를 적용한 y라벨값의 비교
print('기존라벨과 kmeas의 비교 acc:',acc) 

ic(acc) # acc: 0.8933333333333333

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(datasets.data,kmeans.labels_)
score=model.score(datasets.data,kmeans.labels_)
ic(score)

# acc: 0.8933333333333333
# ic| acc: 0.8933333333333333
# ic| score: 1.0
