from tensorflow.python.keras.models import Sequential ,Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Input
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print(x_train.shape,y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

x_train=x_train.reshape(50000, 32, 32, 3)
x_test=x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)


#2. 모델구성
# model=Sequential()
# model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(32,32,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,(3,3),padding='valid', activation='relu'))
# model.add(Conv2D(16,(2,2),activation='relu')) 
# model.add(Flatten())
# model.add(Dense(32,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(10,activation='softmax'))

input1=Input(shape=(32,32,3))
Conv2D1=Conv2D(filters=32, kernel_size=(3,3),activation='relu')(input1)
# Conv2D2=Conv2D(MaxPooling2D())(Conv2D1)
Conv2D2=Conv2D(filters=64, kernel_size=(2,2),activation='relu')(Conv2D1)
Conv2D3=Conv2D(filters=128, kernel_size=(2,2),activation='relu')(Conv2D2)
Conv2D4=Conv2D(filters=64, kernel_size=(2,2),activation='relu')(Conv2D3)
dense1= Flatten()(Conv2D4)
dense2= Dense(32,activation='relu')(dense1)
dense3= Dense(32,activation='relu')(dense2)
output1=Dense(10,activation='softmax')(dense3)
model=Model(inputs=input1,outputs=output1)



# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=100, batch_size=100, verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_test,axis=1)
y_test=np.argmax(y_test,axis=1)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
print('============================')
acc=accuracy_score(y_test,y_predict)
print('acc score :', acc)

# loss :  7.87142276763916
# accuracy :  0.47920000553131104
# ============================
# acc score : 1.0

# loss :  1.2900276184082031
# accuracy :  0.5523999929428101
# ============================
# acc score : 1.0