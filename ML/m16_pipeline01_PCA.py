import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets=load_iris()
x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=1004)

# scaler=MinMaxScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline #소문자니까 함수같아 보이지?
from sklearn.decomposition import PCA

model=make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())
#                  스케일러 두개도 적용 가능하다. (두개 한다고 성능이 좋아지지는 않음 걍 그럼)
#               민맥스로 스케일링하고 , 4개의 컬럼을 2개로 줄이고, 랜덤포레스트로 돌린다.

# 3. 훈련
model.fit(x_train,y_train)
# 파이프라인으로 fit하면 fit_transform 으로 돌아가면서 스케일링이 같이 적용된다.

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

print('--------------------------------------')
y_predict=model.predict(x_test)
acc=accuracy_score(y_test, y_predict)
print('acc_score : ',acc)



# load_iris 
# LinearSVC 결과:  0.8333333333333334
# LinearSVC score : 0.8333333333333334
# =========================================
# LogisticRegression 결과:  0.9333333333333333
# LogisticRegression_acc score : 0.9333333333333333
# =========================================
# KNeighborsClassifier 결과:  0.9
# KNeighborsClassifier_acc score : 0.9
# =========================================
# DecisionTreeClassifier 결과:  0.9333333333333333
# DecisionTreeClassifier_acc score : 0.9333333333333333
# =========================================
# RandomForestClassifier 결과:  0.9666666666666667
# RandomForestClassifier_acc score : 0.9666666666666667

# model.score: 1.0 <- pipeline

