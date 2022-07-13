from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)  #(10000, 28, 28) (10000,)

x_train=x_train.reshape(60000, 28, 28, 1)
x_test=x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,
        train_size=0.8,shuffle=True, random_state=42)

#2. 모델구성
# model=Sequential()
# model.add(Conv2D(filters=50,kernel_size=(6,6),
#                  padding='same',
#                  input_shape=(28,28,1)))
# # model.add(MaxPooling2D())
# model.add(Conv2D(70,(4,4),
#                  padding='valid', 
#                  activation='relu'))
# model.add(Conv2D(40,(2,2),
#                  padding='valid',
#                  activation='relu')) 
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(10,activation='softmax'))

input1=Input(shape=(28,28,1))
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

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='val_loss',patience=50,mode='auto',
                            verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=10, batch_size=100, verbose=1)

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

# loss :  0.3299350440502167
# accuracy :  0.8786666393280029
# ============================
# acc score : 1.0

# loss :  0.382591187953949
# accuracy :  0.8914999961853027
# ============================
# acc score : 1.0