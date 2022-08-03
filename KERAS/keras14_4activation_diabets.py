from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

datasets=load_diabetes()
x=datasets.data
y=datasets.target

# print(x)
# print(y)
# print(x.shape,y.shape) #(442, 10) (442,)

# print(datasets.feature_names)
# print(datasets.DESCR)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=72)

#2. 모델구성
model=Sequential()
model.add(Dense(50,activation='relu',input_dim=10))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='linear'))
model.add(Dense(50,activation='linear'))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=3000, batch_size=100, verbose=1)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


'''
loss: 1080.822998046875
r2스코어: 0.746747006658822

loss: 2614.564697265625
r2스코어: 0.5228523401688681 <-액티베이션 추가 왜...때문에... + 노드추가

loss: 2609.2978515625
r2스코어: 0.561964641082486 <-랜덤스테이트 변경 / 노드 수 변경 

loss: 2467.640869140625
r2스코어: 0.5857452554526845 <-레이어 변경

loss: 2286.047119140625
r2스코어: 0.6537980446009815 <-초기 노드, 레이어, 랜덤스테이트로 변경

loss: 2281.5791015625
r2스코어: 0.6544747250535046 <-노드 수 확 늘임

loss: 2190.218017578125
r2스코어: 0.6683105364383615 <-노드 수 조정

loss: 2271.77880859375
r2스코어: 0.6559588499054116 <-레이어 조정









'''
