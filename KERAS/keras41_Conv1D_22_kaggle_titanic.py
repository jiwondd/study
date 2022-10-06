import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Conv1D,Flatten,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tqdm import tqdm_notebook
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.keras.utils import to_categorical


#1.데이터
path = './_data/kaggle_titanic/' 
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
# print(train_set) # [891 rows x 11 columns]
# print(train_set.describe())
# print(train_set.info())


# print(test_set) # [418 rows x 10 columns]
# print(train_set.isnull().sum()) 

# print(test_set.isnull().sum())


drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

# print(train_set) 
# print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
x = train_set.drop(['Survived'],axis=1) 
# print(x) #(891, 9)
y = train_set['Survived']
# print(y.shape) #(891,)


# test_set.drop(drop_cols, axis = 1, inplace =True)
gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
# y의 라벨값 : (array([0, 1], dtype=int64), array([549, 342], dtype=int64))


# y_class = pd.get_dummies((y))
# print(y_class.shape) # (891, 2)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8,shuffle=True ,random_state=31)

print(x_train.shape, x_test.shape) #(712, 9) (179, 9)

# scaler=MinMaxScaler()
scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_set=scaler.transform(test_set)

x_train = x_train.reshape(712, 9, 1)
x_test = x_test.reshape(179, 9, 1)


#2. 모델 구성
model=Sequential()
model.add(Conv1D(10,2,input_shape=(9,1)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation='linear'))
model.add(Dense(1,activation='sigmoid'))


#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=100, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2)



#4.  평가,예측
y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  
# print(y_predict) 
# print(y_test.shape) #(134,)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)


y_summit = model.predict(test_set)

gender_submission['Survived'] = y_summit
submission = gender_submission.fillna(gender_submission.mean())
submission [(submission <0.5)] = 0  
submission [(submission >=0.5)] = 1  
submission = submission.astype(int)
print('타이타닉_끝났다리')


# acc 스코어 : 0.7094972067039106 <-기존

# acc 스코어 : 0.7653631284916201 <- MinMax

# acc 스코어 : 0.7988826815642458 <-Standard

# acc 스코어 : 0.7541899441340782  <-MaxAbsScaler

# acc 스코어 : 0.7597765363128491  <-RobustScaler

# acc 스코어 : 0.7821229050279329 <- dnn-cnn

# acc 스코어 : 0.7374301675977654 <-LSTM

# acc 스코어 : 0.7821229050279329 <-Conv1D
