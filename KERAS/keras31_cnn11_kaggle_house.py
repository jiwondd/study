import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler


#1. 데이터
path = './_data/kaggle_house/' 
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'sample_submission.csv',
                       index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


print(test_set)
print(train_set.shape) #(1460,76)
print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다. 예측할때 다시 씀

print(train_set.columns)
print(train_set.info()) 
print(train_set.describe()) 

print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.89, shuffle = True, random_state = 68)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) 
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(y)
print(y.shape) # (1460,)
print(x_train.shape) #(1299, 75)
print(x_test.shape) #(161, 75)

x_train = x_train.reshape(1299, 25,3,1)
x_test = x_test.reshape(161, 25,3,1)



#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1, 1),                                
                 padding='same',
                 input_shape=(25, 3,1)))                                                  
model.add(Conv2D(32, (1,1), 
                 padding = 'same',        
                 activation= 'swish'))   
model.add(Conv2D(64, (1,1), 
                 padding = 'same',        
                 activation= 'swish'))   
model.add(Flatten())  
model.add(Dense(100,input_dim=75))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))



# #3. 컴파일,훈련

earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='auto', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='mae', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500, batch_size=100, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2)

            
#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# loss : 18022.71484375
# r2스코어 : 0.8715662226572666
