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

# x=x.reshape (96,4,1)
# print(x.shape) (96, 4)


# 2. 모델구성
model=Sequential()                                                   
model.add(Dense(64,input_dim=4))
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
# x_pred=ccc.reshape(7,4,1)
result=model.predict(ccc)
print('loss:',loss)
print('[96,97,98,99]의 결과:',result)


'''
 
loss: 0.0030851662158966064            <-LSTM
[96,97,98,99]의 결과: [[100.00636]
 [100.8052 ]
 [101.56046]
 [102.1834 ]
 [102.66319]
 [103.07639]
 [103.41372]]
 
loss: 3.344646029290743e-05            <-DNN
[96,97,98,99]의 결과: [[100.00712 ]
 [101.00724 ]
 [102.00736 ]
 [103.007484]
 [104.00764 ]
 [105.00775 ]
 [106.00788 ]]
 
 
데이터가 작을 경우에는 LSTM이 DNN보다 성능이 별로 일 수 있다. 
지금처럼 데이터가 작을 경우에는 시계열 데이터여도 DNN으로 바꿔서 쓸 수 있는데
그래도 왠만하면 시계열 데이터는 LSTM을 쓰는게 낫다
'''
