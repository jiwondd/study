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
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

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

n_splits=5
kfold=KFold(n_splits=n_splits, shuffle=True, random_state=66)


# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)

#2. 모델구성
allAlgorithms=all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model=algorithm()
        scores=cross_val_score(model,x,y,cv=kfold)
    
        print(name,'의 R2:',scores)
        print('cross_val_score:', round(np.mean(scores),4))
    except:
        # continue
        print(name,'은 안나온 놈!!!')
        
# ARDRegression 의 R2: [0.85741523 0.83313939 0.86658892 0.88204685 0.84784806]
# cross_val_score: 0.8574
# AdaBoostRegressor 의 R2: [0.83361776 0.78666965 0.83071025 0.83957496 0.8461959 ]
# cross_val_score: 0.8274
# BaggingRegressor 의 R2: [0.87436771 0.82860732 0.85511812 0.85409788 0.8552668 ]
# cross_val_score: 0.8535
# BayesianRidge 의 R2: [0.85773517 0.83336023 0.86643384 0.88233405 0.84816428]
# cross_val_score: 0.8576
# CCA 의 R2: [ 0.19499371  0.16535224  0.20114741  0.18602278 -0.08560933]
# cross_val_score: 0.1324
# DecisionTreeRegressor 의 R2: [0.73111953 0.77950774 0.74486957 0.79721284 0.77139552]
# cross_val_score: 0.7648
# DummyRegressor 의 R2: [-5.56628733e-04 -4.42446634e-05 -5.40064202e-04 -3.74462486e-04
#  -1.24526191e-03]
# cross_val_score: -0.0006
# ElasticNet 의 R2: [0.8159164  0.79976844 0.80775546 0.82693185 0.81605699]
# cross_val_score: 0.8133
# ElasticNetCV 의 R2: [0.10598662 0.10500645 0.10340937 0.10594503 0.11118982]
# cross_val_score: 0.1063
# ExtraTreeRegressor 의 R2: [0.76689645 0.71644439 0.76537023 0.69047279 0.77993452]
# cross_val_score: 0.7438
# ExtraTreesRegressor 의 R2: [0.86081298 0.82885656 0.88839017 0.87044207 0.86689216]
# cross_val_score: 0.8631
# GammaRegressor 의 R2: [0.79289145 0.75258527 0.7435651  0.77973435 0.77876212]
# cross_val_score: 0.7695
# GaussianProcessRegressor 의 R2: [ -7931.76151771  -6943.80396466 -14147.69832426  -3870.78012345
#   -4553.41486165]
# cross_val_score: -7489.4918
# GradientBoostingRegressor 의 R2: [0.88885154 0.85713102 0.8909658  0.89529185 0.86487575]
# cross_val_score: 0.8794
# HistGradientBoostingRegressor 의 R2: [0.86955914 0.85414648 0.87470418 0.88244052 0.85762858]
# cross_val_score: 0.8677
# HuberRegressor 의 R2: [0.85682162 0.83101842 0.85834264 0.87622147 0.85011959]
# cross_val_score: 0.8545
# IsotonicRegression 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# KNeighborsRegressor 의 R2: [0.85890672 0.84785535 0.8740346  0.86979211 0.84231261]
# cross_val_score: 0.8586
# KernelRidge 의 R2: [0.29622455 0.13175623 0.19927081 0.28208707 0.17183422]
# cross_val_score: 0.2162
# Lars 의 R2: [0.85734087 0.83317929 0.86646217 0.88258861 0.84824313]
# cross_val_score: 0.8576
# LarsCV 의 R2: [0.85780285 0.83324138 0.86646217 0.8822958  0.84565162]
# cross_val_score: 0.8571
# Lasso 의 R2: [0.85734941 0.83318044 0.86646105 0.88258215 0.84824015]
# cross_val_score: 0.8576
# LassoCV 의 R2: [0.85791245 0.83320867 0.86640933 0.88230118 0.84811267]
# cross_val_score: 0.8576
# LassoLars 의 R2: [0.85745833 0.83321191 0.86644198 0.88247888 0.84820518]
# cross_val_score: 0.8576
# LassoLarsCV 의 R2: [0.85780285 0.83324138 0.86646217 0.8822958  0.84814858]
# cross_val_score: 0.8576
# LassoLarsIC 의 R2: [0.85853002 0.83233874 0.86646217 0.88225255 0.84708925]
# cross_val_score: 0.8573
# LinearRegression 의 R2: [0.85734087 0.83317929 0.86646217 0.88258861 0.84824313]
# cross_val_score: 0.8576
# LinearSVR 의 R2: [-6.47323439 -7.08634456 -6.52389082 -6.06756493 -7.69976317]
# cross_val_score: -6.7702
# MLPRegressor 의 R2: [-6.5695823  -7.19946672 -6.62565017 -6.15678945 -7.82168166]
# cross_val_score: -6.8746
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskElasticNetCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLasso 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# MultiTaskLassoCV 의 R2: [nan nan nan nan nan]
# cross_val_score: nan
# NuSVR 의 R2: [-0.0007763  -0.00411133 -0.00332769 -0.00882151 -0.0149156 ]
# cross_val_score: -0.0064
# OrthogonalMatchingPursuit 의 R2: [0.6234925  0.64686279 0.65825844 0.66586761 0.68494581]      
# cross_val_score: 0.6559
# OrthogonalMatchingPursuitCV 의 R2: [0.81829813 0.80323942 0.84560491 0.85253856 0.82138477]
# cross_val_score: 0.8282
# PLSCanonical 의 R2: [-1.94308463 -1.96725245 -1.67466613 -1.58829847 -2.50477454]
# cross_val_score: -1.9356
# PLSRegression 의 R2: [0.86251742 0.83082714 0.85180346 0.87139542 0.83222826]
# cross_val_score: 0.8498
# PassiveAggressiveRegressor 의 R2: [0.85828618 0.81543166 0.8482628  0.86697609 0.84761623]
# cross_val_score: 0.8473
# PoissonRegressor 의 R2: [0.88687753 0.86397609 0.88740087 0.90306534 0.86904615]
# cross_val_score: 0.8821
# RANSACRegressor 의 R2: [0.81745906 0.77551669 0.79594311 0.8423736  0.83526575]
# cross_val_score: 0.8133
# RadiusNeighborsRegressor 의 R2: [-3.54230960e+27 -4.38794307e+27 -4.72789782e+27 -4.21332514e+27
#  -4.16263874e+27]
# cross_val_score: -4.206822875370496e+27
# RandomForestRegressor 의 R2: [0.87382021 0.85186039 0.87426614 0.88018113 0.87373006]
# cross_val_score: 0.8708
# RegressorChain 은 안나온 놈!!!
# Ridge 의 R2: [0.85749548 0.8332507  0.86645664 0.88250369 0.84821761]
# cross_val_score: 0.8576
# RidgeCV 의 R2: [0.85855415 0.83372811 0.86621315 0.88169564 0.84789786]
# cross_val_score: 0.8576
# SGDRegressor 의 R2: [0.8592773  0.83242386 0.8656311  0.88224258 0.84807011]
# cross_val_score: 0.8575
# SVR 의 R2: [-0.02452265 -0.04536184 -0.03673586 -0.04663322 -0.06843935]
# cross_val_score: -0.0443
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 R2: [0.85791009 0.82439937 0.85455544 0.87172579 0.84821702]
# cross_val_score: 0.8514
# TransformedTargetRegressor 의 R2: [0.85734087 0.83317929 0.86646217 0.88258861 0.84824313]     
# cross_val_score: 0.8576
# TweedieRegressor 의 R2: [0.76369052 0.75183531 0.75140361 0.76950808 0.77515883]
# cross_val_score: 0.7623
# VotingRegressor 은 안나온 놈!!!