from graphviz import view
from torchvision.datasets import CIFAR100
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
train_dataset=CIFAR100(path,train=True,download=False) #,transform='' <-스케일링
test_dataset=CIFAR100(path,train=False,download=False)

x_train,y_train=train_dataset.data/255. , train_dataset.targets
x_test,y_test=test_dataset.data/255. , test_dataset.targets

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train).to(DEVICE)
x_test=torch.FloatTensor(x_test)
y_test=torch.LongTensor(y_test).to(DEVICE)
print(x_train.shape,x_test.size()) 

# x_train,x_test=x_train.reshape(50000,3,32,32),x_test.reshape(10000,3,32,32)
x_train,x_test=x_train.permute(0,3,1,2),x_test.permute(0,3,1,2)
print(x_train.shape,x_test.size()) 
# torch.Size([50000, 3, 32, 32]) torch.Size([10000, 3, 32, 32])

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
            nn.Dropout(0.2)) 
        
        self.hidden_layer3=nn.Flatten()
        self.output_layer=nn.Linear(in_features=32*7*7, out_features=100)
        
    def forward(self,x):
        x=self.hidden_layer1(x)
        x=self.hidden_layer2(x)
        x=x.view(x.shape[0],-1)
        x=self.hidden_layer3(x)
        # x=self.hidden_layer4(x)
        # x=self.hidden_layer5(x)
        x=self.output_layer(x)
        return x
        
model=CNN(3).to(DEVICE)
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
epoch:1,loss:4.116425,acc0.087332,val_loss:3.827101,val_acc:0.132288
epoch:2,loss:3.686676,acc0.159609,val_loss:3.615278,val_acc:0.166334
epoch:3,loss:3.494283,acc0.194518,val_loss:3.461469,val_acc:0.203275
epoch:4,loss:3.345147,acc0.222129,val_loss:3.346757,val_acc:0.224241
epoch:5,loss:3.223773,acc0.244742,val_loss:3.256066,val_acc:0.239716
epoch:6,loss:3.125961,acc0.262016,val_loss:3.197737,val_acc:0.251697
epoch:7,loss:3.047342,acc0.275892,val_loss:3.128783,val_acc:0.259585
epoch:8,loss:2.979753,acc0.290507,val_loss:3.065652,val_acc:0.277756
epoch:9,loss:2.919458,acc0.301563,val_loss:3.032802,val_acc:0.280950
epoch:10,loss:2.866762,acc0.313680,val_loss:2.998329,val_acc:0.285543
epoch:11,loss:2.816138,acc0.322197,val_loss:2.978990,val_acc:0.292532
epoch:12,loss:2.772645,acc0.334193,val_loss:2.933015,val_acc:0.300619
epoch:13,loss:2.727306,acc0.342051,val_loss:2.914844,val_acc:0.303115
epoch:14,loss:2.687158,acc0.350108,val_loss:2.902273,val_acc:0.308806
epoch:15,loss:2.646963,acc0.360685,val_loss:2.867843,val_acc:0.311002
epoch:16,loss:2.611202,acc0.366483,val_loss:2.840782,val_acc:0.319389
epoch:17,loss:2.573818,acc0.372821,val_loss:2.833458,val_acc:0.320587
epoch:18,loss:2.540420,acc0.379679,val_loss:2.828594,val_acc:0.319888
epoch:19,loss:2.508104,acc0.386316,val_loss:2.800928,val_acc:0.327875
epoch:20,loss:2.477415,acc0.391815,val_loss:2.783306,val_acc:0.327676
'''