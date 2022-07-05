import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time
gpus=tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    print('지피유돈다룰루')
    aaa='gpu'
else:
    print('지피유안도라유유')
    aaa='cpu'
#1. 데이터
datasets=load_breast_cancer()
print(datasets.feature_names)
print(datasets.DESCR) #(569,30)

x = datasets.data # = x=datasets['data]
y = datasets.target
print(x.shape,y.shape) #(569, 30) (569,) imput 30/output 1
print(x)
print(y)

#2. 모델구성
x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=777)

model=Sequential()
model.add(Dense(300,input_dim=30))
model.add(Dense(400,activation='relu')) 
model.add(Dense(500,activation='relu')) #렐루는 중간 레이어에서만 가능함 / 정확도 80퍼이상
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=200,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
start_time=time.time()
# **이진분류에서 로스는 binary_crossentropy를 사용한다**  
# R2는 회귀모델에서 쓰이니까 분류모델에서는 다른 평가지표를 사용한다 metrics=['accuracy']
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=100, batch_size=100, verbose=1)
end_time=time.time()-start_time

#4. 평가, 예측
loss,acc=model.evaluate(x_test,y_test)
print("loss : ",loss)
y_predict=model.predict(x_test)

# y_predict = model.predict(x_test)
# # print(x_test.shape)


# y_predict[(y_predict<0.5)]=0  
# y_predict[(y_predict>=0.5)]=1  

# print(y_predict)
# print(y_predict.shape)


# acc = accuracy_score(y_test, y_predict)
print('acc score :', acc)
print(aaa,"걸린시간:",end_time)
# [과제] accuracy 값을 완성하자
# r2=r2_score(y_test,y_predict)
# ㄴR2는 회귀모델에 쓰이는 평가지표니까 안쓴다!


'''
loss :  [0.1798676997423172, 0.9122806787490845]
acc score : 0.8508771929824561 
loss :  [0.15738441050052643, 0.9298245906829834]
acc score : 0.9298245614035088 -> 노드갯수 인풋딤보다 2배늘림, 액티베이션 마지막빼고 렐루
loss :  [0.154058575630188, 0.9385964870452881]
acc score : 0.9385964912280702 -> 마지막이랑 바로 위에 액티베이션 시그모이드 나머지 렐루
loss :  [0.17140837013721466, 0.9473684430122375]
acc score : 0.9473684210526315 -> 같은조건으로 한번 더 
'''

# loss :  0.6389638185501099
# acc score : 0.6666666865348816
# cpu 걸린시간: 3.8760018348693848

# loss :  0.6379406452178955
# acc score : 0.6666666865348816
# gpu 걸린시간: 4.954053640365601



