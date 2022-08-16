import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)



#2. 모델구성

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression #리그레션인데 회귀아니고 분류임 어그로...(논리적인회귀=분류)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #여기까지는 보통 잘 안쓰니까 일단은 디폴트로 파라미터로 가보자 
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline 
model=make_pipeline(MinMaxScaler(),RandomForestClassifier())

#3. 컴파일, 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 


# loss :  [0.17140837013721466, 0.9473684430122375]
# acc score : 0.9473684210526315 

# LinearSVC 결과:  0.956140350877193
# LinearSVC score : 0.956140350877193
# =========================================
# LogisticRegression 결과:  0.9649122807017544
# LogisticRegression_acc score : 0.9649122807017544
# =========================================
# KNeighborsClassifier 결과:  0.9649122807017544
# KNeighborsClassifier_acc score : 0.9649122807017544
# =========================================
# DecisionTreeClassifier 결과:  0.9122807017543859
# DecisionTreeClassifier_acc score : 0.9122807017543859
# =========================================
# RandomForestClassifier 결과:  0.9473684210526315
# RandomForestClassifier_acc score : 0.9473684210526315

# model.score: 0.9385964912280702 <-pipeline 사용