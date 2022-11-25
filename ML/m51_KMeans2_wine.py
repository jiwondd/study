from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from icecream import ic

datasets=load_wine()
df=pd.DataFrame(datasets.data,columns=[datasets.feature_names])
# print(df) [178 rows x 13 columns]

kmeans=KMeans(n_clusters=3,random_state=1004)
#             ㄴ라벨의 갯수
kmeans.fit(df)

print(kmeans.labels_) #kmeans적용한 y라벨보기
print(datasets.target) #원래 y라벨보기

df['cluster']=kmeans.labels_
df['target']=datasets.target

acc=accuracy_score(kmeans.labels_,datasets.target)
# 실제 y값과 kmeans를 적용한 y라벨값의 비교
print('acc:',acc) 
ic(acc) # acc: 0.8933333333333333

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(datasets.data,kmeans.labels_)
score=model.score(datasets.data,kmeans.labels_)
ic(score)

# acc: 0.702247191011236
# ic| acc: 0.702247191011236
# ic| score: 1.0
