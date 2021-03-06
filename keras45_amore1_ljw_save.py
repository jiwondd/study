import numpy as np
import pandas as pd
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)
dataset_amo = dataset_amo.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

dataset_sam = dataset_sam.fillna(0)
dataset_amo = dataset_amo.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
dataset_amo = dataset_amo.loc[dataset_amo['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
print(dataset_amo.shape, dataset_sam.shape) # (1035, 11) (1035, 11)

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_amo = dataset_amo.sort_values(by=['일자'], axis=0, ascending=True)

feature_cols = ['시가', '고가', '저가', '거래량', '기관', '외국계', '종가']
label_cols = ['시가']

# 시계열 데이터 만드는 함수
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

SIZE = 20
x1 = split_x(dataset_amo[feature_cols], SIZE)
x2 = split_x(dataset_sam[feature_cols], SIZE)
y = split_x(dataset_amo[label_cols], SIZE)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) # (812, 20, 7) (204, 20, 7)
print(x2_train.shape, x2_test.shape) # (812, 20, 7) (204, 20, 7)
print(y_train.shape, y_test.shape) # (812, 20, 1) (204, 20, 1)

x1_train = x1_train.reshape(812*20,7)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(204*20,7)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(812*20,7)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(204*20,7)
x2_test = scaler.transform(x2_test)

x1_train = x1_train.reshape(812, 20, 7)
x1_test = x1_test.reshape(204, 20, 7)
x2_train = x2_train.reshape(812, 20, 7)
x2_test = x2_test.reshape(204, 20, 7)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(20, 7))
dense1 = Conv1D(64, 2, activation='relu')(input1)
dense2 = LSTM(128, activation='relu')(dense1)
dense3 = Dense(256, activation='relu')(dense2)
dense3 = Dense(128, activation='relu')(dense2)
output1 = Dense(64, activation='relu')(dense3)

# 2-2. 모델2
input2 = Input(shape=(20, 7))
dense11 = Conv1D(64, 2, activation='relu')(input2)
dense12 = LSTM(128, activation='relu')(dense11)
dense13 = Dense(256, activation='relu')(dense12)
dense14 = Dense(128, activation='relu')(dense13)
output2 = Dense(64, activation='relu')(dense14)

merge1 = concatenate([output1, output2])
merge2 = Dense(256, activation='relu')(merge1)
merge3 = Dense(128)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])

# 3. 컴파일, 훈련
import datetime
date=datetime.datetime.now()
print(date)
date=date.strftime('%m%d_%H%M')
print(date)

model.compile(loss='mse', optimizer='adam')
filepath='./_k24/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
mcp=ModelCheckpoint (monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True, 
                    filepath="".join([filepath,'k24_',date,'_','amore_siga',filename]))
model.fit([x1_train, x2_train], y_train, epochs=2000, batch_size=1024, callbacks=[Es,mcp], validation_split=0.2)
model.save("./_test/keras46_amore1_lij_save.h5")

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:])

# loss:  271473280.0
# prdict:  [[131205.66]]
