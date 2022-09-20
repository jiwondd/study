from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:',torch.__version__,'사용DEVICE:',DEVICE)

# 1. 데이터
datasets=load_digits()
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

# print(x_train.size()) #torch.Size([1257, 64])
# print(x_train.shape) 

from torch.utils.data import TensorDataset,DataLoader
train_set=TensorDataset(x_train,y_train) #x,y 합치기
test_set=TensorDataset(x_test,y_test) #x,y 합치기

train_loader=DataLoader(train_set,batch_size=30,shuffle=True)
test_loader=DataLoader(test_set,batch_size=30,shuffle=True)

# 2. 모델
# model=nn.Sequential(
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,10),
#     nn.Softmax()
#     ).to(DEVICE)

class Model(nn.Module): #nn모듈을 상속시키겠다.
    def __init__(self,input_dim,output_dim):
        super().__init__() 
        # super(Model,self).__init__() 위에꺼랑 똑같은거임
        
        self.linear1=nn.Linear(input_dim,64)
        self.linear2=nn.Linear(64,32)
        self.linear3=nn.Linear(32,16)
        self.linear4=nn.Linear(16,output_dim)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax()
        
    def forward(self,input_size): #포워드=순전파
        x=self.linear1(input_size)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        x=self.relu(x)
        x=self.linear4(x)
        x=self.softmax(x)
        return x
        
model=Model(64,10).to(DEVICE)

# 3. 컴파일 훈련
criterion=nn.CrossEntropyLoss() #BCE=Binary Cross Entropy
optimizer=optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,loader):
    total_loss=0
    for x_batch,y_batch in loader:
        
        optimizer.zero_grad()
        hypothesis=model(x_batch)
        loss=criterion(hypothesis,y_batch)
        loss.backward()
        total_loss += loss.item()
    return total_loss /len(loader)

epochs=200
for epoch in range(1, epochs+1):
    loss=train(model,criterion,optimizer,train_loader)
    print('epoch:{},loss:{:8f}'.format(epoch,loss))
    
# 4. 평가 예측
print("===============평가,예측===============")
def evaluate(model,criterion,loader):
    model.eval()
    total_loss=0
    for x_batch,y_batch in loader:
        with torch.no_grad():
            hypothesis=model(x_batch)
            loss=criterion(hypothesis,y_batch)
            total_loss += loss.item()
    return total_loss
    
loss=evaluate(model,criterion,test_loader)
print('loss:',loss)

from sklearn.metrics import accuracy_score
y_pred=torch.argmax(model(x_test),axis=1)
score=accuracy_score(y_test.cpu(),y_pred.cpu())
print('accuracy_score:',score)

# loss: 1.505401372909546
# accuracy_score: 0.9592592592592593

# loss: 1.4968745708465576
# accuracy_score: 0.9629629629629629

# loss: 41.44177985191345
# accuracy_score: 0.10555555555555556