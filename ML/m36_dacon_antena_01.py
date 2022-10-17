import pandas as pd
import random
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split

# 랜덤고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)

# 1. 데이터
path='./_data/dacon_antena/'
train_df = pd.read_csv(path+'train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

# 2. 모델구성,훈련
LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
print('Done.')

# 4. 평가
result=LR.score(train_x,train_y)
print('model.score:',result)


# 5. submission
test_x = pd.read_csv(path+'test.csv').drop(columns=['ID'])

preds = LR.predict(test_x)
print('Done.')

submit = pd.read_csv(path+'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv('./submit.csv', index=False)