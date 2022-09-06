import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten,MaxPool2D, Input,Dropout
from keras.utils import to_categorical

# 1. 데이터
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,28*28).astype('float32')/255.
x_test=x_test.reshape(10000,28*28).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Model=Sequential()

# 2. 모델
def build_model(drop=0.5,optimizer='adam',activation='relu'):
    inputs=Input(shape=(28*28),name='input')
    x=Dense(512,activation=activation, name='hidden1')(inputs)
    x=Dropout(drop)(x)
    x=Dense(256,activation=activation, name='hidden2')(x)
    x=Dropout(drop)(x)
    x=Dense(128,activation=activation, name='hidden3')(x)
    x=Dropout(drop)(x)
    outputs=Dense(10,activation='softmax',name='output')(x)
    
    model=Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'],
                  loass='categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs=[100,200,300,400,500]
    optimizer=['adam','rmsprop','adadelta'] 
    dropout=[0.3,0.4,0.5]
    activation=['relu','linear','sigmoid','selu','elu']
    return{'batch_size':batchs,'optimizer':optimizer,
           'drop':dropout,'activation':activation}
    
hyperparameters=create_hyperparameter()
# print(hyperparameter)
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'],
# 'drop': [0.3, 0.4, 0.5], 'activation': ['relu', 'linear', 'sigmoid', 'selu', 'elu']}

# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

# # model=GridSearchCV(build_model(),hyperparameters, cv=3)
# # model.fit(x_train,y_train)
# # TypeError: estimator should be an estimator implementing 'fit' method, <function build_model at 0x0000024D8BEBB820> was passed    
# # 그리드서치는 스킷런모델인데 위에는 다 케라스이기때문에 안들어간다. 스킷런모델로 바꿔줘야한다.

# keras_model=KerasClassifier(bulid_fn=build_model, verbose=1)
# model=GridSearchCV(keras_model, hyperparameters, cv=3)
# model.fit(x_train,y_train)


from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = GridSearchCV(keras_model, hyperparameters, cv=3)

import time
start = time.time()
model.fit(x_train, y_train, epochs=7, validation_split=0.4)
end = time.time() - start
