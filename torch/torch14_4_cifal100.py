from torchvision.datasets import CIFAR100
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr
# transf=tr.Compose([tr.Resize(150),tr.ToTensor()])
# 리스트 니까 2개이상들어가지?

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

path='./_data/torch_data/'

train_dataset=CIFAR100(path,train=True,download=False) #,transform='' <-스케일링
test_dataset=CIFAR100(path,train=False,download=False)
# 대문자니까 -> 클래스 / 생성자 실행

x_train,y_train=train_dataset.data/255. , train_dataset.targets
x_test,y_test=test_dataset.data/255. , test_dataset.targets

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train).to(DEVICE)
x_test=torch.FloatTensor(x_test)
y_test=torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape,x_test.size())
print(y_train.shape,y_test.size())
# torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])
# torch.Size([50000]) torch.Size([10000])

# print(np.min(x_train.numpy()),np.max(x_train.numpy()))  #0.0 1.0
# x_train,x_test=np.array(x_train),np.array(x_test)

x_train,x_test=x_train.reshape(50000,32*32*3),x_test.reshape(10000,32*32*3)
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
            nn.ReLU(),
            nn.Dropout(0.2))
        self.hidden_layer2=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.hidden_layer3=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.2))            
        self.hidden_layer4=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.hidden_layer5=nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        self.output_layer=nn.Linear(100,100)
        
    def forward(self,x):
        x=self.hidden_layer1(x)
        x=self.hidden_layer2(x)
        x=self.hidden_layer3(x)
        x=self.hidden_layer4(x)
        x=self.hidden_layer5(x)
        x=self.output_layer(x)
        return x
    
    
model=DNN(x_train.shape[1]).to(DEVICE)

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
# 사실 hist라고 하기는 좀 그렇고... loss, acc르 반환하라는 거야 아무튼

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
epoch:1,loss:4.587091,acc0.012476,val_loss:4.495320,val_acc:0.019669
epoch:2,loss:4.226265,acc0.049004,val_loss:4.077978,val_acc:0.071286
epoch:3,loss:4.033119,acc0.072357,val_loss:4.002608,val_acc:0.079373
epoch:4,loss:3.971774,acc0.080934,val_loss:3.959713,val_acc:0.084665
epoch:5,loss:3.934166,acc0.085933,val_loss:3.934484,val_acc:0.087061
epoch:6,loss:3.903395,acc0.089751,val_loss:3.903908,val_acc:0.093850
epoch:7,loss:3.871752,acc0.094950,val_loss:3.892860,val_acc:0.094549
epoch:8,loss:3.825231,acc0.105386,val_loss:3.816547,val_acc:0.115016
epoch:9,loss:3.767612,acc0.116143,val_loss:3.775303,val_acc:0.117312
epoch:10,loss:3.721067,acc0.124560,val_loss:3.733565,val_acc:0.125599
epoch:11,loss:3.679723,acc0.132038,val_loss:3.703796,val_acc:0.133287
epoch:12,loss:3.644038,acc0.137776,val_loss:3.658079,val_acc:0.137280
epoch:13,loss:3.608070,acc0.143754,val_loss:3.646131,val_acc:0.137780
epoch:14,loss:3.575818,acc0.149952,val_loss:3.617541,val_acc:0.144868
epoch:15,loss:3.545642,acc0.155330,val_loss:3.582325,val_acc:0.148962
epoch:16,loss:3.520271,acc0.158289,val_loss:3.560390,val_acc:0.161142
epoch:17,loss:3.495096,acc0.164067,val_loss:3.535403,val_acc:0.162939
epoch:18,loss:3.470204,acc0.170286,val_loss:3.532813,val_acc:0.168630
epoch:19,loss:3.447532,acc0.174224,val_loss:3.507145,val_acc:0.169928
epoch:20,loss:3.423375,acc0.178023,val_loss:3.491521,val_acc:0.169429
'''