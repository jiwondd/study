from sklearn.datasets import load_diabetes
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

# 1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets['target']

x=torch.FloatTensor(x)
y=torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=123,shuffle=True)

# print(x_train.shape,y_train.shape) torch.Size([398, 30]) torch.Size([398])

x_train=torch.FloatTensor(x_train)
y_train=torch.FloatTensor(y_train).to(DEVICE)
x_test=torch.FloatTensor(x_test)
y_test=torch.FloatTensor(y_test).to(DEVICE)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=torch.FloatTensor(x_train).to(DEVICE)
x_test=torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size()) #torch.Size([309, 10])
# print(x_train.shape) 

# 2. 모델
model=nn.Sequential(
    nn.Linear(10,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,1),
    ).to(DEVICE)

# 3. 컴파일 훈련
criterion=nn.MSELoss() #BCE=Binary Cross Entropy
optimizer=optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis=model(x_train)
    loss=criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=500
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
    
loss=evaluate(model,criterion,x_test,y_test)
print('loss:',loss)
# loss: 6155.412109375

from sklearn.metrics import r2_score
y_pred=torch.argmax(model(x_test),axis=1)
score=r2_score(y_test.cpu(),y_pred.cpu())
print('r2_score:',score)

# loss: 5975.50146484375
# r2_score: -3.7178931254725933