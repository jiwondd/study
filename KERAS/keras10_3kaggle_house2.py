#https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

import numpy as np
import datetime as dt 
import pandas as pd
from collections import Counter
import datetime as dt
from sqlalchemy import asc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint

#.1 데이터
path='./_data/kaggle_house/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!
# print(train_set.shape) #(1460, 81)
# print(test_set.shape) #(1459, 80)

#numerical 수치형과 / categorial 범주형 피쳐 나누기
numerical_feats=train_set.dtypes[train_set.dtypes !='object'].index
# print('Number of Numerical features:',len(numerical_feats)) #38
categorial_feats=train_set.dtypes[train_set.dtypes =='object'].index
# print('Number of Categorial features:',len(categorial_feats)) #43

print(train_set[numerical_feats].columns)
print('*'*80)  # *로 줄 나눠라
print(train_set[categorial_feats].columns)

#이상치 확인 / 제거
def detect_outliers(df, n, features):
    outlier_indics=[]
    for col in features:
        Q1=np.percentile(df[col],25) 
        Q3=np.percentile(df[col],75)
        IQR=Q3-Q1
        
        outlier_step=1.5*IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indics.extend(outlier_list_col)
    outlier_indics=Counter(outlier_indics)
    multiple_outliers=list(k for k, v in outlier_indics.items() if v>n)
    
    return multiple_outliers
outliers_to_drop=detect_outliers(train_set, 2, ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',       
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',  
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
       'MiscVal', 'MoSold', 'YrSold', 'SalePrice'])

train_set.loc[outliers_to_drop]

#이상치들 DROP하기
train_set=train_set.drop(outliers_to_drop, axis=0).reset_index(drop=True)
# print(train_set.shape) #(1326, 81)


missing=train_set.isnull().sum()
missing=missing[missing>0]
missing.sort_values(inplace=True)


#내가 모타는...영역...헿 (블로거가 시각화 한 그래프 비교해서 일일이 나눈거임)
num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars',
                   'FullBath','YearBuilt','YearRemodAdd']

num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',
                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]

#결측데이터 처리하기#
# "있다, 없다" 의 개념일 뿐 측정되지 않은 데이터의 의미가 아니다 

cols_fillna=['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

# 그냥 nan으로 두면 비어있다고 오해 할 수 있으니 없다는 의미의 none으로 바꿔준다.
for col in cols_fillna : 
    train_set[col].fillna('None', inplace=True)
    test_set[col].fillna('None', inplace=True)
    
# 결측치의 처리정도를 확인해보자
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])

#남아있는 결측치는 수치형 변수이므로, 평균값으로 대체하자
train_set.fillna(train_set.mean(), inplace=True)
test_set.fillna(test_set.mean(), inplace=True)

#다시한번 확인해보면 (0 0)으로 결측치가 다 처리 되어있는 예쁜모습
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum()) #(0 0)

#'SalePrice'와의 상관관계가 약한 모든 변수를 삭제한다.
id_test=test_set['Id']
to_drop_num=num_weak_corr
to_drop_catg=catg_weak_corr

cols_to_drop=['Id']+to_drop_num+to_drop_catg

for df in [train_set,test_set]:
    df.drop(cols_to_drop,inplace=True,axis=1)

#수치형 변환을 위해 Violinplot과 SalePrice_Log 평균을 참고하여 각 변수들의 범주들을 그룹화 합니다.
#(뭔말인지 모름/ 블로그 카피함ㅎ)
# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 


# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']

#각 범주별로 수치형 변환을 실행합니다. (블로거따라함)
for df in [train_set,test_set]:
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4 

#기존 범주형 변수와 새로 만들어진 수치형 변수 역시 유의하지 않은 것들 삭제하기
train_set.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)
test_set.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)

####   x와 y정의하기   ####
x = train_set.drop('SalePrice',axis=1)
# print(x)
# print(x.columns)
print(x.shape) #(1326, 12)

y = train_set['SalePrice']
# print(y)
print(y.shape) #(1326, )

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=31)

#2. 모델구성
model=Sequential()
model.add(Dense(36,input_dim=12,activation='PReLU'))
model.add(Dense(80,activation='PReLU'))
model.add(Dense(100,activation='PReLU'))
model.add(Dense(90,activation='PReLU'))
model.add(Dense(60,activation='PReLU'))
# model.add(BatchNormalization()) 
model.add(Dense(30,activation='PReLU'))
# model.add(Dropout(0.3))
model.add(Dense(1))

#3. 컴파일, 훈련
# es = EarlyStopping(monitor='val_loss', mode='min')
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train,epochs=500,batch_size=10)
# model.fit(x_train, y_train, nb_epoch=10, batch_size=10, verbose=2, validation_split=0.2)


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("RMSE",rmse)

result = pd.read_csv(path + 'sample_submission.csv', index_col=0)
#index_col=0 의 의미 : index col을 없애준다.

y_summit = model.predict(test_set)
#print(y_summit)
#print(y_summit.shape) # (715,1)


result = pd.read_csv(path + 'sample_submission.csv', index_col=0)
y_summit = model.predict(test_set)
#print(y_summit)
#print(y_summit.shape) # (715,1)

result['SalePrice'] = y_summit

result = abs(result)
result.to_csv(path + 'sample_submission.csv', index=True)






'''
loss: 501670784.0
RMSE 22398.008232956836 -> 0.166

에포 600에서 300으로 변경
loss: 1121406848.0
RMSE 33487.41348678787 

에포 300에서 500으로 변경 / 트레인사이즈 0.99->0.79
loss: 774824704.0
RMSE 27835.673304959237 ->0.17330 

다시 에포 600
loss: 716475264.0
RMSE 26767.05512716395 ->0.16751

랜덤스테이트 777->21
loss: 629375808.0
RMSE 25087.36389269877 ->0.17241

에포600->500 트렌사이즈 0.9
loss: 825812224.0
RMSE 28736.94696104652

액티베이션 selu
loss: 955151360.0
RMSE 30905.526789093816

액티베이션 다시 ReLU/ 랜덤스테이트 777/ 에포 300
loss: 995212352.0
RMSE 31546.987924834808

랜덤스테이트 다시 21
loss: 1728556032.0
RMSE 41575.90743825768 -> 0.23852

트레인사이즈 0.7 / 에포 300->600
loss: 635893568.0
RMSE 25216.930457040216

트레인사이즈 0.79 / 에포 500
loss: 844137984.0
RMSE 29054.052792769256

트레인사이즈 0.99 / 에포 600
loss: 765996736.0
RMSE 27676.645152740657

트레인사이즈 0.89 / 랜덤스테이트 31 
loss: 522000288.0
RMSE 22847.325679951206 ->0.17302

에포600->300
loss: 864219264.0
RMSE 29397.60677491068

에포 300->500
loss: 476288608.0
RMSE 21824.036596565402 -> 0.16979

트레인사이즈 0.5 / 에포 300 / 배치사이즈 10
loss: 625573120.0
RMSE 25011.459202239963

트레인 0.8 / 에포 500
loss: 415214464.0
RMSE 20376.81187852945 -> 0.15973

에포 600 으로 늘려봄
loss: 412587520.0
RMSE 20312.250950276037 ->0.15599

트레인 0.88
loss: 411326528.0
RMSE 20281.18683617204 ->0.15834

에포 600->800
loss: 530420544.0
RMSE 23030.861001325182

에포 800->300
loss: 430325088.0
RMSE 20744.279476863183

트레인 88->99 / 에포 600
loss: 421597280.0
RMSE 20532.834082585545  

에포 600->300
loss: 402627264.0
RMSE 20065.57491933878 ->0.16422

그대로 한번 더
loss: 367946304.0
RMSE 19181.924926211916 ->0.16062

트레인 99->89
loss: 451740288.0
RMSE 21254.18312273537

에포 300->600
loss: 457061152.0
RMSE 21378.986942271036

트레인 89->88 / 에포 600
loss: 458048096.0
RMSE 21402.05983238127

에포600->1000
loss: 513163584.0
RMSE 22653.113549326496

1000->500
loss: 440599392.0
RMSE 20990.459289574737

트레인사이즈 99로
loss: 524874016.0
RMSE 22910.13031881344

트레인 88 에포 600
loss: 516048640.0
RMSE 22716.703967229423

-----------------------------------------------------------
loss: 412553856.0
RMSE 20311.42197541379

액티베이션 swish
loss: 433716128.0
RMSE 20825.852730519102

트레인사이즈 88->99
loss: 568235584.0
RMSE 23837.69334325672

트레인 89 / 
loss: 442543904.0
RMSE 21036.72651102668

액티베이션 sigmoid로 바꾸고 드롭아웃추가
loss: 32051625984.0
RMSE 179029.66831025484 ->5.51684

액티베이션 다시 Relu / 에포800
loss: 7676824064.0
RMSE 87617.48570808432 ->0.67041

드롭아웃빼고 에포 500
loss: 482352704.0
RMSE 21962.52984416479 

시그모이드+BatchNormalization
loss: 31455113216.0
RMSE 177355.9024464291

다시 ReLU / 에포800/ 트레인 88
loss: 1015569042112512.0
RMSE 31867994.887359962 ->0.62052

배치 노말라이제이션 빼고 에포 500
loss: 433423200.0
RMSE 20818.816550377

액티베이션 elu
loss: 411524192.0
RMSE 20286.05766729715 ->0.16986

트레인 0.9 / 랜덤스테이트 777/ 에포 500
loss: 600782464.0
RMSE 24510.864839348436

트레인 0.8 / 랜덤스테이트 31/ 액티베이션 relu
loss: 515624896.0
RMSE 22707.375519886085

에포 800
loss: 438561280.0
RMSE 20941.855277980492

노드 수 확 늘려봄
loss: 400826272.0
RMSE 20020.646238975976 ->0.15804

에포 3000번
loss: 544884480.0
RMSE 23342.761092496403

에포 5000번
loss: 546509376.0
RMSE 23377.54158156522

액티베이션 prelu / 에포 500
loss: 457385216.0
RMSE 21386.564905098006

에포만 800
loss: 495715584.0
RMSE 22264.67209665359

에포만 500
loss: 613269312.0
RMSE 24764.275041100333

에포 600
loss: 411551584.0
RMSE 20286.733549643526

노드 확 늘리고 에포1000
loss: 444911840.0
RMSE 21092.933931043943

노드 다시 줄이고 에포500
loss: 542246144.0
RMSE 23286.17806011016

노드 조금씩 변경
loss: 444530400.0
RMSE 21083.889821469176


'''
