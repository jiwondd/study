import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split, KFold

path='./_data/dacon_veggie/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

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
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) #, False)

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)

x_train, x_test, y_train, y_test = train_test_split(
    train_data,label_data, train_size=0.8, shuffle=True, random_state=123)



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
import time

model=Sequential()
model.add(Conv1D(64,2,input_shape=(1440,37)))
model.add(Flatten())
model.add(Dense(128,activation='swish'))
model.add(Dense(64,activation='swish'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

earlyStopping=EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,restore_best_weights=True) 
start=time.time()
model.compile(loss='mse',optimizer="adam")
hist=model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
               epochs=1000, batch_size=100, verbose=1)
end=time.time()

loss=model.evaluate(x_test,y_test)
print('loss:',loss)
y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('걸린시간 :',np.round(end-start,2))

import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()

# loss: 900967369277440.0
# r2스코어: -561333412258936.94
# 걸린시간 : 47.14