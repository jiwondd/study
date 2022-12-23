import tensorflow as tf
from keras.datasets import mnist
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
# from tensorflow.python.keras.optimizer_v2.adam import Adam # tf 2.7 ver은 이러캐 해야함
from keras.optimizers import Adam # tf 2.9 ver은 이러캐 해야함
from sklearn.metrics import accuracy_score

print(kt.__version__) #1.1.3
print(tf.__version__) #2.9.1

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255. ,x_test/255.


class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(
                units=hp.Int("units1", min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation1',values=['relu','selu','elu'])))
        model.add(Dropout(hp.Choice('dropout1',values =[0.0, 0.2, 0.3, 0.4, 0.5])))
        
        model.add(Dense(10, activation="softmax"))
        
        model.compile(
            optimizer=Adam(lr = hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])), 
            loss="sparse_categorical_crossentropy", metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32]),
            **kwargs,
        )

kerastuner = kt.RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)
# kerastuner = kt.Hyperband(get_model,
#                           directory = 'my_dir',
#                           objective = 'val_acc',
#                           max_epochs = 6,
#                           project_name = 'kerastuner-mnist2')

kerastuner.search(x_train,y_train,
                  validation_data=(x_test,y_test),
                  epochs=5)
best_hps = kerastuner.get_best_hyperparameters(num_trials=2)[0]

print('best parameter - units1 : ',best_hps.get('units1'))
# print('best parameter - units2 : ',best_hps.get('units2'))
# print('best parameter - units3 : ',best_hps.get('units3'))
# print('best parameter - units4 : ',best_hps.get('units4'))

print('best parameter - dropout1 : ',best_hps.get('dropout1'))
# print('best parameter - dropout2 : ',best_hps.get('dropout2'))
# print('best parameter - dropout3 : ',best_hps.get('dropout3'))
# print('best parameter - dropout4 : ',best_hps.get('dropout4'))

print('best parameter - activation1 : ',best_hps.get('activation1'))
# print('best parameter - activation2 : ',best_hps.get('activation2'))
# print('best parameter - activation3 : ',best_hps.get('activation3'))
# print('best parameter - activation4 : ',best_hps.get('activation4'))

print('best parameter - learning_rate : ',best_hps.get('learning_rate'))

# Best val_accuracy So Far: 0.9672999978065491    
# Total elapsed time: 00h 02m 37s
# best parameter - units1 :  224
# best parameter - dropout1 :  0.5
# best parameter - activation1 :  selu
# best parameter - learning_rate :  0.005





    

    