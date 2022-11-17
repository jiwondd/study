from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. 데이터
datasets=load_breast_cancer()
x,y=datasets.data,datasets.target
# print(x.shape,y.shape) (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=123, train_size=0.8, shuffle=True,stratify=y
)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. 모델구성
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression #이름은 리그레션인데 "이진"분류에 쓴다 (시그모이드로 결과를 뺀다)
model=BaggingClassifier(LogisticRegression(),
                        n_estimators=100,
                        n_jobs=-1,
                        random_state=123
                        )

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
print(model.score(x_test,y_test)) #0.9736842105263158
