from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#1. 데이터 
datasets=load_iris()
# print(datasets.DESCR)
#    :Number of Instances: 150 (50 in each of three classes) / 150개의 행
#    :Number of Attributes: 4 numeric, predictive attributes and the class /4개의 컬럼, 열, 피쳐, 특성
# print(datasets.feature_names)
x=datasets['data']
y=datasets.target
# print(x)
# print(x.shape) #(150, 4)
# print(y)
# print(y.shape) #(150,)


# tensorflow.kersa에서의 one-hot-encoding
# print('y의 라벨값:',np.unique(y)) #   y의 라벨값: [0 1 2]
y=to_categorical(y)
# print(y)
# print(y.shape) #(150, 3)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# # 2. 모델구성
model=Sequential()
model.add(Dense(20,input_dim=4))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='linear'))
model.add(Dense(3,activation='softmax')) #softmax로 나온 결과 값은 다 합치면 1이다. 

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
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','iris',filename]))

hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping,mcp],
               epochs=100, batch_size=100, verbose=1)


# 4. 평가, 예측

# 1.첫번째 방법
# loss,acc=model.evaluate(x_test,y_test)
# print("loss : ",loss)
# print('acc score :', acc)


# 2.두번재 방법
result=model.evaluate(x_test,y_test)
print('loss:',result[0])
print('accuracy:',result[1])
# print("===================================")
# print(y_test[:5])
# print("===================================")
# y_pred=model.predict(x_test[:5])
# print(y_pred)
print("===================================")

y_predict=model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
# print(y_predict)
# [1 2 0 1 2 0 2 1 0 0 2 1 2 0 2 1 1 1 2 0 2 2 0 2 1 0 1 1 1 1]
y_test=np.argmax(y_test,axis=1)
# print(y_test)
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)


'''
loss: 0.18647046387195587
accuracy: 0.9166666865348816
===================================
acc score : 1.0

'''