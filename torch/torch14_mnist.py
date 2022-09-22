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
train_dataset=MNIST(path,train=True,download=True) #,transform='' <-스케일링
test_dataset=MNIST(path,train=False,download=True)

# 대문자니까 -> 클래스 / 생성자 실행

x_train,y_train=train_dataset.data/255. , train_dataset.targets
x_test,y_test=test_dataset.data/255. , test_dataset.targets

print(x_train.shape,x_test.size())
print(y_train.shape,y_test.size())
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
# torch.Size([60000]) torch.Size([10000])
# print(train_dataset[0][0].shape) #torch.Size([1, 15, 15]) 위에서 리사이즈 했으니까

# print(np.min(x_train))
# min() received an invalid combination of arguments - got (axis=NoneType, out=NoneType, ), but expected one of:
# 위에 토치텐서형태로 되어있으니까 넘파이가 안먹힌다.

print(np.min(x_train.numpy()),np.max(x_train.numpy()))  #0.0 1.0

x_train,x_test=x_train.view(-1,28*28),x_test.view(-1,28*28)
print(x_train.shape,x_test.size())

train_dset=TensorDataset(x_train,y_train)
test_dset=TensorDataset(x_test,y_test)
train_loader=DataLoader(train_dset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dset,batch_size=32,shuffle=False)

class DNN(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        
        self.hidden_layer1=nn.Sequential(
            nn.Linear(num_features,100),
            nn.ReLU())
        self.hidden_layer2=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU())
        self.hidden_layer3=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU())            
        self.hidden_layer4=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU())
        self.hidden_layer5=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU())
        self.output_layer=nn.Linear(100,10)
        
    def forward(self,x):
        x=self.hidden_layer1(x)
        x=self.hidden_layer2(x)
        x=self.hidden_layer3(x)
        x=self.hidden_layer4(x)
        x=self.hidden_layer5(x)
        x=self.output_layer(x)
        return x
    
    
model=DNN(784).to(DEVICE)

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
    model.eval()
    epoch_loss=0
    epoch_acc=0
    
    with torch.no_grad():
        for x_batch,y_batch in loader:
            x_batch,y_batch=x_batch.to(DEVICE),y_batch.to(DEVICE)
            
            hypothesis=model(x_batch)
            loss=criterion(hypothesis,y_batch)

            y_pred=torch.argmax(hypothesis,1)
            acc=(y_pred==y_batch).float().mean()
            
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