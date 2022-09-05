from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. 데이터
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)
#reshape했을때 서로의 곱한 값이 같아야 한다,순서가 섞이면 안된다. 

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

print(x_train.shape) #(60000, 28, 28, 1)
print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)


#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=10,kernel_size=(4,4),
                 padding='same',
                 input_shape=(28,28,1)))  ##(batch_size, row, column, channels)   
model.add(MaxPooling2D()) #맥스풀링 "레이어"는 아니고 위에 붙어있는 놈이다. 
model.add(Conv2D(8,(3,3),
                 padding='valid', 
                 activation='relu'))
model.add(Conv2D(6,(2,2),
                 padding='valid',
                 activation='relu')) 
model.add(Flatten()) #(N, 63)
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))


from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

learning_rate = 0.001

optlist = [adam.Adam, adadelta.Adadelta, adagrad.Adagrad, adamax.Adamax, rmsprop.RMSprop, nadam.Nadam]
result=[]
for optimizer in optlist:
    optimizer = optimizer(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=0)

    # 4. evaluate, predict
    loss = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred=np.argmax(y_pred,axis=1)
    y_pred=to_categorical(y_pred)
    acc=accuracy_score(y_pred,y_test)
    re='optimizer: ', optimizer.__class__.__name__, '/loss: ', round(loss, 4), '/lr: ', learning_rate, '/acc', acc
    result.append(re)
print(re)

# loss :  0.23986844718456268
# accuracy :  0.9788333177566528

# optimizer:  Adam    /loss:  0.1343 /lr:  0.001 /acc 0.9830833333333333






