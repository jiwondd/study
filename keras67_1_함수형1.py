from keras.models import Model
from keras.layers import Dense, Flatten, Input, Dropout, GlobalAvgPool2D
from keras.applications import VGG16
from keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

input = Input(shape = (32, 32, 3))
vgg16 = VGG16(include_top= False)(input)
gap = GlobalAvgPool2D()(vgg16)
hidden1 = Dense(1287)(gap)
hidden2 = Dense(64, activation = 'relu')(hidden1)
hidden3 = Dense(32)(hidden2)
output1 = Dense(100, activation = 'softmax')(hidden3)

model = Model(inputs = input, outputs = output1)
model.trainable=False

model.compile(optimizer='adam', metrics=['acc'],loss='sparse_categorical_crossentropy')

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4,batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('걸린시간 : ', end - start)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))

# 걸린시간 :  495.4595732688904
# loss :  27.0389
# accuracy :  0.0005