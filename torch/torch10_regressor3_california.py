from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

# 1. 데이터
datasets=fetch_california_housing()
x=datasets.data
y=datasets['target']

x=torch.FloatTensor(x)
y=torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=123,shuffle=True)

# print(x_train.shape,y_train.shape) torch.Size([398, 30]) torch.Size([398])

x_train=torch.FloatTensor(x_train)
y_train=torch.FloatTensor(y_train).to(DEVICE)
x_test=torch.FloatTensor(x_test)
y_test=torch.FloatTensor(y_test).to(DEVICE)

# scaler=StandardScaler()
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=torch.FloatTensor(x_train).to(DEVICE)
x_test=torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size()) #torch.Size([14447, 8])
# print(x_train.shape) 

# 2. 모델
model=nn.Sequential(
    nn.Linear(8,128),
    nn.ReLU(),
    nn.Linear(128,64),
    # nn.Sigmoid(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,1),
    ).to(DEVICE)

# 3. 컴파일 훈련
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis=model(x_train)
    loss=criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=5000
for epoch in range(1, epochs+1):
    loss=train(model,criterion,optimizer,x_train,y_train)
    print('epoch:{},loss:{:8f}'.format(epoch,loss))
    
# 4. 평가 예측
print("===============평가,예측===============")
def evaluate(model,criterion,x_test,y_test):
    model.eval()
    with torch.no_grad():
        y_pred=model(x_test)
        loss=criterion(y_pred,y_test)
    return loss.item()
    
from sklearn.metrics import r2_score
loss=evaluate(model,criterion,x_test,y_test)

# y_pred=torch.argmax(model(x_test),axis=1)
# score=r2_score(y_test.cpu(),y_pred.cpu())
# print('loss:',loss)
# print('r2_score:',score)

y_pred=model(x_test).cpu().detach().numpy()
score=r2_score(y_pred,y_test.cpu().detach().numpy())
print('loss:',loss)
print('r2_score:',score)

# loss: 1.323022723197937
# r2_score: -3.1873629180378185