import tensorflow as tf
from keras.datasets import mnist
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
# from tensorflow.python.keras.optimizer_v2.adam import Adam # tf 2.7 ver은 이러캐 해야함
from keras.optimizers import Adam # tf 2.9 ver은 이러캐 해야함


print(kt.__version__) #1.1.3
print(tf.__version__) #2.9.1

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test = x_train/255., x_test/255.

def get_model(hp):
    hp_unit1=hp.Int('unit1',min_value=16,max_value=512,step=16)
    hp_unit2=hp.Int('unit2',min_value=16,max_value=512,step=16)
    hp_unit3=hp.Int('unit3',min_value=16,max_value=512,step=16)
    hp_unit4=hp.Int('unit4',min_value=16,max_value=512,step=16)
    
    hp_drop1=hp.Choice('dropout',values=[0.0, 0.2, 0.3, 0.4, 0.5])
    hp_drop2=hp.Choice('dropout',values=[0.0, 0.2, 0.3, 0.4, 0.5])
    
    hp_lr=hp.Choice('learning_rate',values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    
    model=Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(hp_unit1, activation='relu'))
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit2, activation='relu'))
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit3, activation='relu'))
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(hp_unit4, activation='relu'))
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(10,activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=hp_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
    
kerastuner=kt.Hyperband(get_model,
                        directory='my_dir',
                        objective='val_accuracy',
                        max_epochs=6,
                        project_name='keras-mnist2')

kerastuner.search(x_train,y_train,
                  validation_data=(x_test,y_test),epochs=5)

best_hps=kerastuner.get_best_hyperparameters(num_trials=2)[0]

print('best parameter - units1 : ',best_hps.get('unit1'))
print('best parameter - units2 : ',best_hps.get('unit2'))
print('best parameter - units3 : ',best_hps.get('unit3'))
print('best parameter - units4 : ',best_hps.get('unit4'))

print('best parameter - dropout1 : ',best_hps.get('dropout1'))
print('best parameter - dropout2 : ',best_hps.get('dropout2'))

##################################################################
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es=EarlyStopping(monitor='val_loss',patience=10, mode='min',verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',factor=0.5,verbose=1)

model=kerastuner.hypermodel.build(best_hps)
history = model.fit(x_train,y_train,
                    validation_data=(x_test,y_test),
                    epochs=30,
                    callbacks=[es,reduce_lr])
loss, accuracy= model.evaluate(x_test,y_test)

print('acc : ',accuracy)
print('loss : ',loss)

y_predict=model.predict(x_test)

# 원핫 안했으니까 얘는 그냥 빼자ㅎ
# from sklearn.metrics import accuracy_score
# acc_score=accuracy_score(y_test,y_predict)
# print('acc_score : ',acc_score)







    

    