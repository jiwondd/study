import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델구성
def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    
    return model

# 반환한 모델에는 컴파일까지 포함

def create_hyperparameters():
    batches = [100, 200, 300, 400, 500] 
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    return {'batch_size' : batches, 'optimizer' : optimizers,
            'drop' : dropout, 'activation' : activation}

# 수치화된 딕셔너리(키, 밸류) 형태로 반환

hyperparameters = create_hyperparameters()
# print(hyperparameters)

# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = GridSearchCV(keras_model, hyperparameters, cv=3)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=1, verbose=1) # iter 디폴트 10 3*5 = 15번 돌림

import time
start = time.time()
model.fit(x_train, y_train, epochs=3, validation_split=0.4)
end = time.time()

print('걸린시간 : ', end - start)
print('model.best_params_ : ', model.best_params_)
print('model.best_estimator_ : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))