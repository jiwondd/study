from torchvision.datasets import CIFAR10
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

train_dataset=CIFAR10(path,train=True,download=True) #,transform='' <-스케일링
test_dataset=CIFAR10(path,train=False,download=True)
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
        
        self.output_layer=nn.Linear(100,10)
        
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
epoch:1,loss:2.125788,acc0.174104,val_loss:1.992091,val_acc:0.237919
epoch:2,loss:1.890567,acc0.299964,val_loss:1.811934,val_acc:0.330471
epoch:3,loss:1.782766,acc0.348349,val_loss:1.770131,val_acc:0.357328
epoch:4,loss:1.708533,acc0.377519,val_loss:1.689555,val_acc:0.391374
epoch:5,loss:1.658050,acc0.398712,val_loss:1.666381,val_acc:0.399062
epoch:6,loss:1.616833,acc0.414967,val_loss:1.593764,val_acc:0.424720
epoch:7,loss:1.580967,acc0.430602,val_loss:1.561518,val_acc:0.437300
epoch:8,loss:1.549797,acc0.440859,val_loss:1.605870,val_acc:0.426218
epoch:9,loss:1.528906,acc0.448177,val_loss:1.520248,val_acc:0.456170
epoch:10,loss:1.506769,acc0.457054,val_loss:1.503806,val_acc:0.462660
epoch:11,loss:1.490334,acc0.463352,val_loss:1.500432,val_acc:0.466254
epoch:12,loss:1.470435,acc0.471029,val_loss:1.492774,val_acc:0.463958
epoch:13,loss:1.458632,acc0.476587,val_loss:1.488048,val_acc:0.468550
epoch:14,loss:1.446921,acc0.480786,val_loss:1.467965,val_acc:0.478634
epoch:15,loss:1.428992,acc0.485565,val_loss:1.478011,val_acc:0.473343
epoch:16,loss:1.418324,acc0.490923,val_loss:1.456366,val_acc:0.479732
epoch:17,loss:1.404804,acc0.496881,val_loss:1.465262,val_acc:0.473542
epoch:18,loss:1.393323,acc0.498680,val_loss:1.461543,val_acc:0.476538
epoch:19,loss:1.382325,acc0.504259,val_loss:1.435165,val_acc:0.490615
epoch:20,loss:1.369666,acc0.509397,val_loss:1.427653,val_acc:0.489117
'''