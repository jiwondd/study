# 1. 데이터
import numpy as np
from sklearn.metrics import r2_score
x1_datasets=np.array([range(100),range(301,401)]) #삼성전자의 주가(0~100), 하이닉스의 종가(301~400)
x2_datasets=np.array([range(101,201), range(411,511), range(150,250)]) #원유, 돈육, 밀
x1=np.transpose(x1_datasets)
x2=np.transpose(x2_datasets)

print(x1_datasets.shape,x2_datasets.shape) #(100, 2) (100, 3)

y=np.array(range(2001,2101)) #금리
from sklearn.model_selection import train_test_split

x1_train, x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1,x2,y,train_size=0.7, shuffle=True, random_state=777)

print(x1_train.shape,x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape,x2_test.shape) #(70, 3) (30, 3)
print(y_train.shape,y_test.shape) #(70,) (30,)

# 2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

# 2-1 모델 1
input1=Input(shape=(2,))
dense1=Dense(32, activation='relu', name='jw1')(input1)
dense2=Dense(64, activation='relu', name='jw2')(dense1)
dense3=Dense(32, activation='relu', name='jw3')(dense2)
output1=Dense(10, activation='relu', name='out_jw1')(dense3)

# 2-2 모델 2
input2=Input(shape=(3,))
densea=Dense(24, activation='relu', name='ajw1')(input2)
denseb=Dense(32, activation='relu', name='bjw2')(densea)
densec=Dense(64, activation='relu', name='cjw3')(denseb)
densed=Dense(32, activation='relu', name='djw3')(densec)
output2=Dense(10, activation='relu', name='out_jw2')(densed)

from tensorflow.python.keras.layers import concatenate, Concatenate #(리스트, 함수)
# merge1=concatenate([output1,output2], name='mg1') # concatenate = 사슬처럼 이어주다 (append랑 비슷하다)
merge1=Concatenate(name='mg1')([output1,output2])
# ㄴ 10개의 노드 + 33개의 노드가 합쳐진 하나의 Dense 모델이 되었어요
merge2=Dense(100,activation='relu',name='mg2')(merge1)
merge3=Dense(100,name='mg3')(merge2)
last_out=Dense(1,name='last')(merge3)
model=Model(inputs=[input1,input2],outputs=last_out)
model.summary()

'''
# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit([x1_train,x2_train],y_train,epochs=1000)

# 4. 평가, 예측
loss=model.evaluate([x1_test,x2_test],y_test)
y_pred=model.predict([x1_test,x2_test])
r2=r2_score(y_test,y_pred)
print('r2스코어:',r2)
print('loss:',loss)

# r2스코어: 0.9990107337236135
# loss: 0.9606929421424866
'''