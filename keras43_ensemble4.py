# 잘 써볼 일 없는 예제입니다! x는 1개 / y는 2개 

# 1. 데이터
import numpy as np
from sklearn.metrics import r2_score
x1_datasets=np.array([range(100),range(301,401)]) #삼성전자의 주가(0~100), 하이닉스의 종가(301~400)
x1=np.transpose(x1_datasets)

print(x1_datasets.shape) #(2, 100)

y1=np.array(range(2001,2101)) #금리
y2=np.array(range(201,301))

from sklearn.model_selection import train_test_split

x1_train, x1_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,y1,y2,train_size=0.7, shuffle=True, random_state=777)

print(x1_train.shape,x1_test.shape) #(70, 2) (30, 2)
print(y1_train.shape,y1_test.shape) #(70,) (30,)
print(y2_train.shape,y2_test.shape) #(70,) (30,)


# 2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

# 2-1 모델 1
input1=Input(shape=(2,))
dense1=Dense(32, activation='relu', name='jw1')(input1)
dense2=Dense(64, activation='relu', name='jw2')(dense1)
dense3=Dense(32, activation='relu', name='jw3')(dense2)
output1=Dense(10, activation='relu', name='out_jw1')(dense3)

# concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate #(리스트, 함수)
merge1=concatenate([output1], name='mg1') # concatenate = 사슬처럼 이어주다 (append랑 비슷하다)
# ㄴ 10개의 노드 + 33개의 노드가 합쳐진 하나의 Dense 모델이 되었어요
merge2=Dense(100,activation='relu',name='mg2')(merge1)
merge3=Dense(100,name='mg3')(merge2)
last_out=Dense(1,name='last1')(merge3)

# 2-4 output 1 
output41=Dense(100,activation='relu',name='mg4')(last_out)
output42=Dense(100,name='mg5')(output41)
last_out2=Dense(1,name='last2')(output42)

# 2-4 output 2
output51=Dense(100,activation='relu',name='mgg4')(last_out)
output52=Dense(100,name='mgg5')(output51)
output53=Dense(100,name='mgg6')(output52)
last_out3=Dense(1,name='last3')(output53)

model=Model(inputs=input1,outputs=[last_out2,last_out3])
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit([x1_train],[y1_train,y2_train],epochs=1000)


# 4. 평가, 예측
loss=model.evaluate(x1_test,[y1_test,y2_test])
# loss2=model.evaluate([x1_test,x2_test,x3_test],y2_test)
y_pred1,y_pred2=model.predict(x1_test)
r2_1=r2_score(y1_test,y_pred1)
r2_2=r2_score(y2_test,y_pred2)
print('y의_loss:',loss)
print('y1의_r2스코어:',r2_1)
print('y2의_r2스코어:',r2_2)


# y의_loss: [779.857666015625, 13.501235008239746, 766.3564453125]
# y1의_r2스코어: 0.9860963718616248
# y2의_r2스코어: 0.21080605071985747
