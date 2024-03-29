from re import I
import pandas as pd
import numpy as np
import glob
import os

from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Conv1D, Flatten,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping


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

# 2. 모델
model = Sequential()
model.add(LSTM(10, input_shape=(1440,37)))
model.add(Dense(6, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit(train_data,label_data, epochs=10, callbacks=[Es], validation_split=0.1)


# 4. 평가, 예측
loss = model.evaluate(vali_data, val_target)
print(loss)
test_pred = model.predict(test_input)
print(test_pred.shape) # (195, 1)



# test1 = pd.read_csv(path+'test_target/TEST_01.csv', index_col=False) # 29
# test1['rate'] = test_pred[:len(test1)][0]
# test1.to_csv(test_target+'test_target/TEST_01.csv', index=False)

# test2 = pd.read_csv(test_target+'test_target/TEST_02.csv', index_col=False) # 35
# test2['rate'] = test_pred[len(test1):len(test1)+len(test2)][0]
# test2.to_csv(test_target+'test_target/TEST_02.csv', index=False)


# test3 = pd.read_csv(test_target+'test_target/TEST_03.csv', index_col=False)
# test3['rate'] = test_pred[30+35:30+35+26][0]
# test3.to_csv(test_target+'test_target/TEST_03.csv', index=False)


# test4 = pd.read_csv(test_target+'test_target/TEST_04.csv', index_col=False)
# test4['rate'] = test_pred[30+35+26:30+35+26+32][0]
# test4.to_csv(test_target+'test_target/TEST_04.csv', index=False)


# test5 = pd.read_csv(test_target+'test_target/TEST_05.csv', index_col=False)
# test5['rate'] = test_pred[30+35+26+32:30+35+26+32+37][0]
# test5.to_csv(test_target+'test_target/TEST_05.csv', index=False)


# test6 = pd.read_csv(test_target+'test_target/TEST_05.csv', index_col=False)
# test6['rate'] = test_pred[30+35+26+32+37:][0]
# test6.to_csv(test_target+'test_target/TEST_06.csv', index=False)


for i in range(6):
    i2=0
    a = i+1
    thisfile = './_data/dacon_veggie/test_target/'+'TEST_0'+str(a)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[i2:i2+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    i2+=len(test['rate'])



import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("./_data/dacon_veggie/test_target/")
with zipfile.ZipFile("submission3.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()