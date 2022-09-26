from graphviz import view
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr
transf=tr.Compose([tr.Resize(150),tr.ToTensor()])
# 리스트 니까 2개이상들어가지?

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

path='./_data/torch_data/'

# train_dataset=MNIST(path,train=True,download=True,transform=transf) #,transform='' <-스케일링
# test_dataset=MNIST(path,train=False,download=True,transform=transf)#처음에만 다운로드 하고 그 뒤로는 다운로드 false해도 되요
train_dataset=MNIST(path,train=True,download=False) #,transform='' <-스케일링
test_dataset=MNIST(path,train=False,download=False)

# 대문자니까 -> 클래스 / 생성자 실행

x_train,y_train=train_dataset.data/255. , train_dataset.targets
x_test,y_test=test_dataset.data/255. , test_dataset.targets

# print(x_train.shape,x_test.size())
# print(y_train.shape,y_test.size())
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
# torch.Size([60000]) torch.Size([10000])
# print(train_dataset[0][0].shape) #torch.Size([1, 15, 15]) 위에서 리사이즈 했으니까

# print(np.min(x_train))
# min() received an invalid combination of arguments - got (axis=NoneType, out=NoneType, ), but expected one of:
# 위에 토치텐서형태로 되어있으니까 넘파이가 안먹힌다.

# print(np.min(x_train.numpy()),np.max(x_train.numpy()))  #0.0 1.0

# 기존에는 n,28,28,1 이었는데 토치에서는 n,1,28,28 채널을 앞으로
# x_train,x_test=x_train.view(-1,28*28),x_test.view(-1,28*28)
x_train,x_test=x_train.unsqueeze(1),x_test.unsqueeze(1)
print(x_train.shape,x_test.size()) #torch.Size([60000, 1, 28, 28]) torch.Size([10000, 1, 28, 28])
# 0은 스칼라 / 1은 벡터 / 2는 매트릭스 / 3 이상은 텐서

train_dset=TensorDataset(x_train,y_train)
test_dset=TensorDataset(x_test,y_test)
train_loader=DataLoader(train_dset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dset,batch_size=32,shuffle=False)

class CNN(nn.Module):
    def __init__(self,num_features):
        super(CNN,self).__init__()
        
        self.hidden_layer1=nn.Sequential(
            nn.Conv2d(num_features,64,kernel_size=(3,3),stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2))
        
        self.hidden_layer2=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2))   # 32, 5, 5
        
        self.hidden_layer3=nn.Linear(32*5*5,32)
    

        # self.hidden_layer3=nn.Sequential(
        #     nn.Linear(100,100),
        #     nn.ReLU(),
        #     nn.Dropout(0.2))     
               
        # self.hidden_layer4=nn.Sequential(
        #     nn.Linear(100,100),
        #     nn.ReLU(),
        #     nn.Dropout(0.2))
        
        # self.hidden_layer5=nn.Sequential(
        #     nn.Linear(100,100),
        #     nn.ReLU(),
        #     nn.Dropout(0.2))
        
        self.output_layer=nn.Linear(in_features=32, out_features=10)
        
    def forward(self,x):
        x=self.hidden_layer1(x)
        x=self.hidden_layer2(x)
        x=x.view(x.shape[0],-1)
        x=self.hidden_layer3(x)
        # x=self.hidden_layer4(x)
        # x=self.hidden_layer5(x)
        x=self.output_layer(x)
        return x
        
model=CNN(1).to(DEVICE)

# 3. 컴파일 훈련
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4) #0.0001

def train(model,criterion,optimizer,loader):
    epoch_loss=0
    epoch_acc=0
    for x_batch, y_batch in loader:
        x_batch,y_batch=x_batch.to(DEVICE),y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis=model(x_batch)
        loss=criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        
        y_pred=torch.argmax(hypothesis,1)
        acc=(y_pred==y_batch).float().mean()
        
        epoch_acc+=acc.item()
        
    return epoch_loss/len(loader),epoch_acc/len(loader)
# hist=model.fit(x_train,y_train) / hist에는 loss,acc
# 사실 hist라고 하기는 좀 그렇고... loss, acc르 반환하라는 거야 아무트

# 4. 평가, 예측
def evaluate(model,criterion,loader):
    model.eval() #위에서 모델을 그래도 가지고 왔어도 이발을 넣어주면서 이밸류에이트를 하게 되니까 드롭아웃이 알아서 빠진다.
    epoch_loss=0
    epoch_acc=0
    
    with torch.no_grad():
        for x_batch,y_batch in loader:
            x_batch,y_batch=x_batch.to(DEVICE),y_batch.to(DEVICE)
            
            hypothesis=model(x_batch)
            loss=criterion(hypothesis,y_batch)

            y_pred=torch.argmax(hypothesis,1)
            acc=(y_pred==y_batch).float().mean()

            epoch_loss+=loss.item()
            epoch_acc+=acc.item()
            
    return epoch_loss/len(loader),epoch_acc/len(loader)       
# loss,acc=model.evaluate(x_test,y_test)

epochs=20
for epoch in range(1, epochs+1):
    loss,acc=train(model,criterion,optimizer,train_loader)
    val_loss,val_acc=evaluate(model,criterion,test_loader)
    print('epoch:{},loss:{:4f},acc{:3f},val_loss:{:4f},val_acc:{:3f}'.format(
        epoch,loss,acc,val_loss,val_acc
    ))
    
'''
torch.Size([60000, 1, 28, 28]) torch.Size([10000, 1, 28, 28])
epoch:1,loss:0.553510,acc0.841750,val_loss:0.170457,val_acc:0.951577
epoch:2,loss:0.143291,acc0.957800,val_loss:0.098183,val_acc:0.970447
epoch:3,loss:0.097426,acc0.970917,val_loss:0.079113,val_acc:0.976138
epoch:4,loss:0.078100,acc0.976150,val_loss:0.062743,val_acc:0.979832
epoch:5,loss:0.066594,acc0.980183,val_loss:0.055820,val_acc:0.982927
epoch:6,loss:0.058896,acc0.982717,val_loss:0.052016,val_acc:0.981929
epoch:7,loss:0.053032,acc0.983783,val_loss:0.050581,val_acc:0.983227
epoch:8,loss:0.048832,acc0.985533,val_loss:0.045098,val_acc:0.984924
epoch:9,loss:0.045392,acc0.986000,val_loss:0.046997,val_acc:0.984625
epoch:10,loss:0.042339,acc0.987100,val_loss:0.041468,val_acc:0.985024
epoch:11,loss:0.039199,acc0.987850,val_loss:0.046527,val_acc:0.985523
epoch:12,loss:0.036814,acc0.988783,val_loss:0.040908,val_acc:0.985823
epoch:13,loss:0.034698,acc0.989333,val_loss:0.037754,val_acc:0.987220
epoch:14,loss:0.032892,acc0.989917,val_loss:0.048644,val_acc:0.983427
epoch:15,loss:0.031555,acc0.990867,val_loss:0.039626,val_acc:0.987220
epoch:16,loss:0.029681,acc0.990800,val_loss:0.043095,val_acc:0.986222
epoch:17,loss:0.028364,acc0.991333,val_loss:0.041076,val_acc:0.986122
epoch:18,loss:0.027084,acc0.991833,val_loss:0.037503,val_acc:0.987420
epoch:19,loss:0.025410,acc0.992283,val_loss:0.036945,val_acc:0.987121
epoch:20,loss:0.024620,acc0.992467,val_loss:0.036996,val_acc:0.987420
'''