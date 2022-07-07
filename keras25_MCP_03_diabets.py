from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=72)


scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# 2. 모델구성
model=Sequential()
model.add(Dense(50,activation='relu',input_dim=10))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='linear'))
model.add(Dense(50,activation='linear'))
model.add(Dense(1))

#3.컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date) #2022-07-07 17:50:42.752072
date=date.strftime('%m%d_%H%M')
print(date) #0707_1750

filepath='./_k24/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'
#        ㄴ4글자로 표시 /로스는 4자리 까지            

earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='auto',
                            verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer="adam")
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','diabets',filename]))

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
               epochs=100, batch_size=100, verbose=1)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)


'''
loss: 2962.46484375
r2스코어: 0.5513604224019597

'''