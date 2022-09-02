from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
import joblib

#.1 데이터
path='./_data/dacon_travel/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)

train_set = train_set.replace({'Gender' : 'Fe Male'}, 'Female')
test_set = test_set.replace({'Gender' : 'Fe Male'}, 'Female')
train_set = train_set.replace({'Occupation':'Free Lancer'}, 'Small Business')
test_set = test_set.replace({'Occupation':'Free Lancer'}, 'Small Business')
train_set = train_set.replace({'MaritalStatus' : 'Divorced'}, 'Single')
test_set = test_set.replace({'MaritalStatus' : 'Divorced'}, 'Single')

train_set.loc[train_set['NumberOfTrips'] != train_set['NumberOfTrips'], 'NumberOfTrips'] = train_set['NumberOfTrips'].fillna(0)
test_set.loc[test_set['NumberOfTrips'] != test_set['NumberOfTrips'], 'NumberOfTrips'] = test_set['NumberOfTrips'].fillna(0)
train_set.loc[train_set['DurationOfPitch'] != train_set['DurationOfPitch'], 'DurationOfPitch'] = train_set['DurationOfPitch'].fillna(0)
test_set.loc[test_set['DurationOfPitch'] != test_set['DurationOfPitch'], 'DurationOfPitch'] = test_set['DurationOfPitch'].fillna(0)

train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)
train_set['DurationOfPitch']=np.round(train_set['DurationOfPitch'],0).astype(int)
test_set['DurationOfPitch']=np.round(test_set['DurationOfPitch'],0).astype(int)
train_set['MonthlyIncome']=np.round(train_set['MonthlyIncome'],0).astype(int)
test_set['MonthlyIncome']=np.round(test_set['MonthlyIncome'],0).astype(int)

train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','NumberOfFollowups'], axis=1)
train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)
# print(total_set.head(10))

total_set = pd.get_dummies(total_set)
# print(total_set.isnull().sum())

# imputer=IterativeImputer(random_state=777)
# imputer.fit(total_set)
# total_set=imputer.transform(total_set)

scaler=QuantileTransformer()
# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(total_set)
x=scaler.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=777,train_size=0.8,stratify=y)

kFold=StratifiedKFold(shuffle=True,random_state=777)

# smote=SMOTE(random_state=777)
# x_train,y_train=smote.fit_resample(x_train,y_train)
# print(np.unique(y, return_counts=True))

# 2. 모델구성
#################################베이지안##########################################
# cb_params={
#     'reg_lambda':(0,6),
#     'bagging_temperature':(0,7),
#     'subsample':(0.5,5),
#     'max_depth':(6,12),
#     'best_model_min_trees':(0,5),
#     'min_data_in_leaf':(0,5),
#     'one_hot_max_size':(0,6),
# }

# def cb_hamsu(reg_lambda,bagging_temperature,subsample,max_depth,
#              best_model_min_trees,min_data_in_leaf,one_hot_max_size):
#     params={
#         'n_estimators':500,'learning_rate':0.02,
#         'sampling_frequency':'PerTree',
#         'reg_lambda':max(reg_lambda,0), #양수만 받는다 0이상
#         'bagging_temperature':max(bagging_temperature,0),
#         'subsample':max(min(subsample,1),0),
#         # 'subsample':max(min(subsample,1),0), # 0~1 사이로
#         'max_depth':int(round(max_depth)), #정수로
#         'best_model_min_trees':int(round(best_model_min_trees)),
#         'min_data_in_leaf':int(round(min_data_in_leaf)),
#         'one_hot_max_size':int(round(one_hot_max_size)), # 10이상의 정수
#     }
#     model=CatBoostClassifier(**params)
#     model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],
#             #   eval_metric='accuracy_score',
#               verbose=0,
#               early_stopping_rounds=50)
#     y_pred=model.predict(x_test)
#     result=accuracy_score(y_test,y_pred)
    
#     return result

# lgb_bo=BayesianOptimization(f=cb_hamsu,
#                             pbounds=cb_params,
#                             random_state=123)

# lgb_bo.maximize(init_points=5,n_iter=50)

# print(lgb_bo.max)

# 'params': {'bagging_temperature': 0.0, 'best_model_min_trees': 5.0, 'max_depth': 12.0,
#             'min_data_in_leaf': 5.0, 'one_hot_max_size': 0.0, 'reg_lambda': 0.8075994717190443,
#             'subsample': 5.0}

# model = CatBoostClassifier(
#     n_estimators = 500, learning_rate = 0.02, sampling_frequency='PerTreeLevel',
#     best_model_min_trees=5, 
#     max_depth = 12,
#     min_data_in_leaf=5,
#     subsample = max(min(5.0, 1), 0), 
#     reg_lambda= 0.8075994717190443,
#     one_hot_max_size=0,
#     bagging_temperature=0
# )

model = CatBoostClassifier(
    n_estimators = 500, 
    sampling_frequency='PerTreeLevel',
    learning_rate = 0.5752401680334096, 
    best_model_min_trees=int(round(1.692337514392721)), 
    max_depth = int(round(11.50198910232388)),
    min_data_in_leaf=4.703607467808818,
    subsample = max(min(4.294621152299014, 1), 0), 
    reg_lambda= 2.9814132186209052,
    one_hot_max_size=int(round(1.499808335054402)),
    bagging_temperature=6.827137620981638
)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
# print('model.score:',result) 

#5. 데이터 summit
model.fit(x,y)
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
print('model.score:',result) 
submission.to_csv('./_data/dacon_travel/sample_submission_5.csv', index=True)

# model.score: 0.8925831202046036
# model.score: 0.8951406649616368 <-전처리 추가 (median)
# model.score: 0.8976982097186701 <- mean
# model.score: 0.9028132992327366 <-fillna(0)
# model.score: 0.9002557544757033 <-스모트넣기
# model.score: 0.9028132992327366 <-스탠다드 스케일
# model.score: 0.9028132992327366 <-RobustScaler
# model.score: 0.8695652173913043 <-뭐고...? (lgb bay)
# model.score: 0.9104859335038363 <-cat bay (에포500)
# model.score: 0.907928388746803 <-cat bay(에포 1000)
# model.score: 0.9104859335038363 <-에포 700
# model.score: 0.9053708439897699 <-러닝레이트 0.3
# model.score: 0.9104859335038363 <-sampling_frequency='PerTreeLevel' / 0.9062233589
# 

