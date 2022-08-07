#https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

import numpy as np
import datetime as dt 
import pandas as pd
from collections import Counter
import datetime as dt
from sqlalchemy import asc
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler

#.1 데이터
path='./_data/kaggle_house/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!

#numerical 수치형과 / categorial 범주형 피쳐 나누기
numerical_feats=train_set.dtypes[train_set.dtypes !='object'].index
# print('Number of Numerical features:',len(numerical_feats)) #38
categorial_feats=train_set.dtypes[train_set.dtypes =='object'].index
# print('Number of Categorial features:',len(categorial_feats)) #43

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
# print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum()) #(0 0)

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
y = train_set['SalePrice']

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.9,shuffle=True, random_state=31)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorithms=all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        model.fit(x_train,y_train)
    
        y_predict=model.predict(x_test)
        r2=r2_score(y_test,y_predict)
        print(name,'의 정답률 : ',r2)
    except:
        # continue
        print(name,'은 안나온 놈!!!')
        
# ARDRegression 의 정답률 :  0.8348399987209832
# AdaBoostRegressor 의 정답률 :  0.8091145559371864
# BaggingRegressor 의 정답률 :  0.8653519955433546
# BayesianRidge 의 정답률 :  0.8358670055077169
# CCA 의 정답률 :  0.020895298710018828
# DecisionTreeRegressor 의 정답률 :  0.7123876844751411
# DummyRegressor 의 정답률 :  -0.009583901805204409
# ElasticNet 의 정답률 :  0.8313517654205208
# ElasticNetCV 의 정답률 :  0.10458330844703945
# ExtraTreeRegressor 의 정답률 :  0.7474556914774617
# ExtraTreesRegressor 의 정답률 :  0.8552482669508263
# GammaRegressor 의 정답률 :  0.8004961099953529
# GaussianProcessRegressor 의 정답률 :  -2958.0469993954716
# GradientBoostingRegressor 의 정답률 :  0.8789184410151718
# HistGradientBoostingRegressor 의 정답률 :  0.8738851710456181
# HuberRegressor 의 정답률 :  0.8389695923153394
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.8760440052183146
# KernelRidge 의 정답률 :  0.06288884181560017
# Lars 의 정답률 :  0.8347353814627427
# LarsCV 의 정답률 :  0.8361221339403586
# Lasso 의 정답률 :  0.8347604219520524
# LassoCV 의 정답률 :  0.8361941554522198
# LassoLars 의 정답률 :  0.8351648866169212
# LassoLarsCV 의 정답률 :  0.8361221339403586
# LassoLarsIC 의 정답률 :  0.8394936484644736
# LinearRegression 의 정답률 :  0.8347353814627424
# LinearSVR 의 정답률 :  -8.210022131625811
# MLPRegressor 의 정답률 :  -8.369358908975716
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.0027950840846002745
# OrthogonalMatchingPursuit 의 정답률 :  0.5013247279301334
# OrthogonalMatchingPursuitCV 의 정답률 :  0.8333169427753091
# PLSCanonical 의 정답률 :  -2.5359769841642312
# PLSRegression 의 정답률 :  0.8488252987523323
# PassiveAggressiveRegressor 의 정답률 :  0.8378095151859132
# PoissonRegressor 의 정답률 :  0.8490202738643862
# RANSACRegressor 의 정답률 :  0.8193210204931186
# RadiusNeighborsRegressor 의 정답률 :  -4.5263102603591183e+27
# RandomForestRegressor 의 정답률 :  0.8802807544536293
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.8351720756455874
# RidgeCV 의 정답률 :  0.8351720756455976
# SGDRegressor 의 정답률 :  0.8361748506919469
# SVR 의 정답률 :  -0.014634013912449895
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.8345604112292456
# TransformedTargetRegressor 의 정답률 :  0.8347353814627424
# TweedieRegressor 의 정답률 :  0.7871872827118652
# VotingRegressor 은 안나온 놈!!!