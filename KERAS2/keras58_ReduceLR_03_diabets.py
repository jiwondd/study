from gc import callbacks
import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

#1. 데이터
datasets=load_diabetes()
x=datasets['data']
y=datasets.target

print(x.shape) #(442, 10)
# print(np.unique(y)) #[0 1]

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


#2. 모델구성
model=Sequential()
model.add(Dense(64,input_dim=10))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(8,activation='relu'))
model.add(Dense(1)) 


from tensorflow.keras.optimizers import Adam

learnig_rate=0.01
optimizer=Adam(lr=learnig_rate)

model.compile(optimizer=optimizer, metrics=['mse'],
                loss='mse')
    
    
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es=EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auteo',verbose=1,
                            factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4,batch_size=128,callbacks=[es,reduce_lr])
end = time.time()

from sklearn.metrics import accuracy_score
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('걸린시간 : ', end - start)
print('loss : ', round(loss,4))

# 걸린시간 :  4.321568965911865
# loss :  2349.7981