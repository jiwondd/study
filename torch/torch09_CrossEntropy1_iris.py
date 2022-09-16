# 논리회귀=2진분류모델 (헷갈리기 쉽다!!)
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

# 1. 데이터
datasets=load_iris()
x=datasets.data
y=datasets['target']

x=torch.FloatTensor(x)
y=torch.LongTensor(y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=123,shuffle=True,stratify=y)

# print(x_train.shape,y_train.shape) torch.Size([398, 30]) torch.Size([398])

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train).to(DEVICE)
x_test=torch.FloatTensor(x_test)
y_test=torch.LongTensor(y_test).to(DEVICE)
# 인트가 길어지면 롱 / 플로트가 길어지면 더블

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=torch.FloatTensor(x_train).to(DEVICE)
x_test=torch.FloatTensor(x_test).to(DEVICE)

# print(x_train.size()) torch.Size([105, 4])
# print(x_train.shape) torch.Size([105, 4])

# 2. 모델
model=nn.Sequential(
    nn.Linear(4,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,3),
    nn.Softmax()
    ).to(DEVICE)

# 3. 컴파일 훈련
criterion=nn.CrossEntropyLoss() #BCE=Binary Cross Entropy
optimizer=optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    hypothesis=model(x_train)
    loss=criterion(hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs=100
for epoch in range(1, epochs+1):
    loss=train(model,criterion,optimizer,x_train,y_train)
    print('epoch:{},loss:{:8f}'.format(epoch,loss))
    
# 4. 평가 예측
print("===============평가,예측===============")
def evaluate(model,criterion,x_test,y_test):
    model.eval()
    with torch.no_grad():
        hypothesis=model(x_test)
        loss=criterion(hypothesis,y_test)
    return loss.item()
    
loss=evaluate(model,criterion,x_test,y_test)
print('loss:',loss)

from sklearn.metrics import accuracy_score
y_pred=torch.argmax(model(x_test),axis=1)
score=accuracy_score(y_test.cpu(),y_pred.cpu())
print('accuracy_score:',score)

# loss: 0.6167663931846619
# accuracy_score: 0.9333333333333333

'''
# acc score 식 만들어보기
y_pred=(model(x_test)>=0.5).float()
score=(y_pred==y_test).float().mean()
print('acc:{:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score=accuracy_score(y_test,y_pred)
# print('accuracy_score:',score)
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

score=accuracy_score(y_test.cpu(),y_pred.cpu())
print('accuracy_score:',score)

# loss: 1.2157402038574219
# acc:0.9825
# accuracy_score: 0.9824561403508771
'''