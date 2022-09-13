import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
#인풋 아웃풋이 같다, 특성이 가장 두드러진 부분이 뽑아져나온다. (얼굴을 넣으면 눈코입이 특성이니까 나오고 주근깨 여드름은 지워짐 좋은데?)

x_train=x_train.reshape(60000, 784).astype('float32')/255.
x_test=x_test.reshape(10000, 784).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img=Input(shape=(784,))

encoded=Dense(64,activation='relu')(input_img)
#             ㄴ이 숫자는 줄일수도 늘일수도 있다. 이 과정에서 중요한 특성은 남고 필요없는 애들은 사라진다.
# 큰노드에서 작은노드 거쳐서 다시 원래대로

# encoded=Dense(1064,activation='relu')(input_img) #노드를 늘여보자 / 족금 좋아진거같기두?
# encoded=Dense(16,activation='relu')(input_img) #노드를 줄여보자 / 그림이 많이 흐릿해졌다

decoded=Dense(784,activation='sigmoid')(encoded)
# 위에서 /255 해줘서 0에서 1사이로 되어있기때문에 시그모이드 적용해줌(다른 데이터에서는 액티베이숀 다른거 넣어도 됨) 
decoded=Dense(784,activation='relu')(encoded)   #액티베이션 렐루로 바꿔보자 / 그림이 많이 지저분해졌다.
decoded=Dense(784,activation='linear')(encoded) #액티베이션 리니어로 바꿔보자 / 왕별로
decoded=Dense(784,activation='tanh')(encoded)   #액티베이션 탄(-1 ~ 1) 바꿔보자 / 걍 뭐
autoencoder=Model(input_img,decoded)
# autoencoder.summary()


# 컴파일 훈련
# autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam',loss='mse')

autoencoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

autoencoder.fit(x_train,x_train,epochs=30,batch_size=128,validation_split=0.2)
# 여기서는 y값이 없어서 훈련시킬때 x값으로 (준지도학습)
decoded_imgs=autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()