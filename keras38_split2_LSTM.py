import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU

# 1. 모델구성 
a=np.array(range(1,101)) # 1부터 101-1 까지 = 총 100개
x_predict=np.array(range(96,106))

size=5 # x는 4개 y는 1개

def split_x(dataset,size):
    aaa=[]
    for i in range(len(dataset)-size+1):
        subset=dataset[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)


bbb=split_x(a,size)
print(bbb)
print(bbb.shape) # (96, 5)

x=bbb[:, :-1]
y=bbb[:, -1]
print(x,y)
print(x.shape,y.shape) # (96, 4) (96,)

ccc=split_x(x_predict,4) # x프레딕트 를 split_x 함수를 이용해서 4개씩 잘라보쟈
print(ccc)
print(ccc.shape) #(7, 4)

x=x.reshape (96,4,1)
print(x.shape)

# [실습] 모델 구성, 평가 예측 할 것! 

# 2. 모델구성
model=Sequential()                                                   
model.add(LSTM(units=64,input_shape=(4,1)))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(326,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000)

# 4. 평가, 예측
loss=model.evaluate(x,y)
x_pred=ccc.reshape(7,4,1)
result=model.predict(x_pred)
print('loss:',loss)
print('[96,97,98,99]의 결과:',result)


''' 
왜 7개 나오냐면 프레딕트가 쉐입이 7,4,1 이니까 7개가 나온거여요 

loss: 0.0011895523639395833
[96,97,98,99]의 결과: [[ 99.84879 ] ->100
 [100.5494  ] ->101
 [100.87963 ] ->102
 [101.129684] ->103
 [101.373985] ->104
 [101.57395 ] ->105
 [101.771645]] ->106
 
loss: 0.0030851662158966064
[96,97,98,99]의 결과: [[100.00636]
 [100.8052 ]
 [101.56046]
 [102.1834 ]
 [102.66319]
 [103.07639]
 [103.41372]]
 
loss: 0.020284617319703102            <-GRU 개구리ㅎ
[96,97,98,99]의 결과: [[ 99.795494]
 [100.50444 ]
 [100.88703 ]
 [101.148155]
 [101.40327 ]
 [101.63624 ]
 [101.8635  ]]
 
 
 
 
 '''