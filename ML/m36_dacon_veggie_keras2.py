import pandas as pd
import numpy as np
import glob
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, GRU, Conv1D, Flatten, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold

path = './_data/dacon_veggie/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))
test_input = sorted(glob.glob(path + 'test_input/*.csv'))
test_target = sorted(glob.glob(path + 'test_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)

train_data, label_data = aaa(train_input_list, train_target_list) #, False)
vali_data, val_target = aaa(val_input_list, val_target_list) #, False)

test_input, test_target = aaa(test_input, test_target)

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 16079
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
print(vali_data.shape) # (206, 1440, 37)


# ######################################################################################################

    
x_train,x_test,y_train,y_test = train_test_split(train_data,label_data,train_size=0.8,shuffle=True,random_state=123)
print(x_train.shape)

#2. 모델 구성      
                                                                                              
model = Sequential()
model.add(LSTM(64,input_shape=(1440,37)))
# model.add(GRU(50, activation='relu'))
# model.add(GRU(50))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1))
# model.summary()

#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from tensorflow.python.keras.optimizers import adam_v2

es = EarlyStopping(monitor='val_loss',patience=400,mode='auto',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=200,mode='auto',verbose=1,factor=0.5)
learning_rate = 0.2
optimizer = adam_v2.Adam(lr=learning_rate)
start_time = time.time()
model.compile(loss='mae', optimizer='adam',metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=2, batch_size=1100, 
                validation_data=(vali_data, val_target),
                verbose=2,callbacks = [es,reduced_lr]
                )
model.save_weights("C:\Study\_save/dacon_vegi02.h5")


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_predict,y_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_predict))
                      
                  
# y_predict = model.predict(x_test)
# print(y_test.shape) #(152,)
# print(y_predict.shape) #(152, 13, 1)

# from sklearn.metrics import accuracy_score, r2_score,accuracy_score
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)

model.fit(train_data,label_data)
y_summit = model.predict(test_target)

path2 = './_data/dacon_veggie\\test_target/' # ".은 현재 폴더"
targetlist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv','TEST_06.csv']
# [29, 35, 26, 32, 37, 36]
empty_list = []
for i in targetlist:
    test_target2 = pd.read_csv(path2+i)
    empty_list.append(test_target2)
    
empty_list[0]['rate'] = y_summit[:29]
empty_list[0].to_csv(path2+'TEST_01.csv')
empty_list[1]['rate'] = y_summit[29:29+35]
empty_list[1].to_csv(path2+'TEST_02.csv')
empty_list[2]['rate'] = y_summit[29+35:29+35+26]
empty_list[2].to_csv(path2+'TEST_03.csv')
empty_list[3]['rate'] = y_summit[29+35+26:29+35+26+32]
empty_list[3].to_csv(path2+'TEST_04.csv')
empty_list[4]['rate'] = y_summit[29+35+26+32:29+35+26+32+37]
empty_list[4].to_csv(path2+'TEST_05.csv')
empty_list[5]['rate'] = y_summit[29+35+26+32+37:]
empty_list[5].to_csv(path2+'TEST_06.csv')
# submission = submission.fillna(submission.mean())
# submission = submission.astype(int)

import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("./_data/dacon_veggie/test_target/")
with zipfile.ZipFile("./_data/dacon_veggie/test_target/keras_sub01.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
print('Done')
print('R2 :', r2)
print('RMSE :', rmse)
end_time = time.time()-start_time
print('걸린 시간:', end_time)
