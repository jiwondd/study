# 1. 데이터
import numpy as np
from sklearn.metrics import r2_score
x1_datasets=np.array([range(100),range(301,401)]) #삼성전자의 주가(0~100), 하이닉스의 종가(301~400)
x2_datasets=np.array([range(101,201), range(411,511), range(150,250)]) #원유, 돈육, 밀
x3_datasets=np.array([range(100,200),range(1301,1401)])
x1=np.transpose(x1_datasets)
x2=np.transpose(x2_datasets)
x3=np.transpose(x2_datasets)

print(x1_datasets.shape,x2_datasets.shape) #(100, 2) (100, 3)

y1=np.array(range(2001,2101)) #금리
y2=np.array(range(201,301))

from sklearn.model_selection import train_test_split

x1_train, x1_test,x2_train,x2_test,x3_train, x3_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,x2,x3,y1,y2,train_size=0.7, shuffle=True, random_state=777)

print(x1_train.shape,x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape,x2_test.shape) #(70, 3) (30, 3)
print(x3_train.shape,x3_test.shape) #(70, 3) (30, 3)
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

# 2-2 모델 2
input2=Input(shape=(3,))
densea=Dense(24, activation='relu', name='ajw1')(input2)
denseb=Dense(32, activation='relu', name='bjw2')(densea)
densec=Dense(64, activation='relu', name='cjw3')(denseb)
densed=Dense(32, activation='relu', name='djw3')(densec)
output2=Dense(10, activation='relu', name='out_jw2')(densed)

# 2-3 모델 3
input3=Input(shape=(3,))
dense11=Dense(24, activation='relu', name='jj1')(input3)
dense22=Dense(32, activation='relu', name='jj2')(dense11)
dense33=Dense(64, activation='relu', name='jj3')(dense22)
dense44=Dense(32, activation='relu', name='jj4')(dense33)
output3=Dense(10, activation='relu', name='out_jw3')(dense44)

# concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate #(리스트, 함수)
merge1=concatenate([output1,output2,output3], name='mg1') # concatenate = 사슬처럼 이어주다 (append랑 비슷하다)
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

model=Model(inputs=[input1,input2,input3],outputs=[last_out2,last_out3])
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train],epochs=1000)


# 4. 평가, 예측
loss=model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
y_pred1,y_pred2=model.predict([x1_test,x2_test,x3_test])
r2_1=r2_score(y1_test,y_pred1)
r2_2=r2_score(y2_test,y_pred2)
print('y의_loss:',loss)
print('y1의_r2스코어:',r2_1)
print('y2의_r2스코어:',r2_2)


# r2스코어: 0.9990107337236135
# loss: 0.9606929421424866

# r2스코어: 0.9995945188766177
# loss: 0.3938042223453522   <- x3추가하기!

# y의_loss: [751.5885620117188, 11.456974029541016, 740.131591796875]
#            ㄴy1+y2          , y1의 로스값        , y2의 로스값
# y1의_r2스코어: 0.9882019036830799
# y2의_r2스코어: 0.23781240963079808
