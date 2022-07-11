from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=72)

print(x_train.shape,x_test.shape) #(353, 10) (89, 10)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train=x_train.reshape(353, 5, 2, 1)
x_test=x_test.reshape(89, 5, 2, 1)


#2. 모델구성
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(1,1),input_shape=(5,2,1)))
model.add(Conv2D(128,kernel_size=(1,1),activation='relu'))
model.add(Conv2D(64,kernel_size=(1,1),activation='relu')) 
model.add(Flatten())               
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=100, batch_size=100, verbose=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)



# loss: 2962.46484375
# r2스코어: 0.5513604224019597

# loss: 3014.39599609375
# r2스코어: 0.5434959293886923 <-dropout적용

# loss: 2347.169677734375
# r2스코어: 0.6445415824815983



