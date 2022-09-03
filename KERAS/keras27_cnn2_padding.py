from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D


model=Sequential()
model.add(Conv2D(filters=10,kernel_size=(2,2),
                 padding='same',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(7,(2,2),
                 padding='valid', #<-이거는 디폴트 값
                 activation='relu')) 
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()             
