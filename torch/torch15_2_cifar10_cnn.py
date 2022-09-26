from graphviz import view
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

path='./_data/torch_data/'

# train_dataset=MNIST(path,train=True,download=True,transform=transf) #,transform='' <-스케일링
# test_dataset=MNIST(path,train=False,download=True,transform=transf)#처음에만 다운로드 하고 그 뒤로는 다운로드 false해도 되요
train_dataset=CIFAR10(path,train=True,download=False) #,transform='' <-스케일링
test_dataset=CIFAR10(path,train=False,download=False)

x_train,y_train=train_dataset.data/255. , train_dataset.targets
x_test,y_test=test_dataset.data/255. , test_dataset.targets

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train).to(DEVICE)
x_test=torch.FloatTensor(x_test)
y_test=torch.LongTensor(y_test).to(DEVICE)
print(x_train.shape,x_test.size()) 

x_train,x_test=x_train.reshape(50000,3,32,32),x_test.reshape(10000,3,32,32)
print(x_train.shape,x_test.size()) 

train_dset=TensorDataset(x_train,y_train)
test_dset=TensorDataset(x_test,y_test)
train_loader=DataLoader(train_dset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dset,batch_size=32,shuffle=False)

class CNN(nn.Module):
    def __init__(self,num_features):
        super(CNN,self).__init__()
        
        self.hidden_layer1=nn.Sequential(
            nn.Conv2d(num_features,64,kernel_size=(2,2),stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2))
        
        self.hidden_layer2=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2))   # 32, 5, 5
        
        self.hidden_layer3=nn.Linear(32*5*5,32)
        
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
print(model)

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

'''