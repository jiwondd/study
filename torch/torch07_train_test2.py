import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn #뉴럴네트워크
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA=torch.cuda.is_available() #쿠다에서 사용가능한 애들을 (대문자)유즈쿠다로 할게
DEVICE=torch.device('cuda'if USE_CUDA else 'cpu') #쿠다를 쓸 수 있르면 쓰고 안되면 씨피유로 할게

#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])
x_predict=np.array([11,12,13])

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,
    train_size=0.7,
    #shuffle=True,
    random_state=66)


x_train=torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE) 
x_test=torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE) 
y_train=torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test=torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_predict=torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)

# x_predict=torch.FloatTensor(np.transpose(x_predict)).to(DEVICE)

print(x_train.shape,y_train.shape,x_predict.shape)
# torch.Size([7, 1]) torch.Size([7, 1]) torch.Size([3])

x_predict=(x_predict-torch.mean(x_train))/torch.std(x_train)
x_test=(x_test-torch.mean(x_train))/torch.std(x_train)
x_train=(x_train-torch.mean(x_train))/torch.std(x_train)# =StandardScaler


# 2.모델구성
model=nn.Sequential(nn.Linear(1,5),
                    nn.Linear(5,3),
                    nn.Linear(3,4),
                    nn.ReLU(),
                    nn.Linear(4,2),
                    nn.Linear(2,1)).to(DEVICE)


# 3.컴파일 훈련
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001) 

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad() 
    
    hypothesis=model(x_train)
    loss=criterion(hypothesis,y_train) # =mse
    
    loss.backward() 
    optimizer.step() 
    return loss.item()

epochs=1500
for epoch in range(1,epochs+1):
    loss=train(model,criterion,optimizer,x_train,y_train)
    print('epoch:{},loss:{}'.format(epoch,loss))
    
    
# 4. 평가, 예측
def evaluate(model,criterion,x_test,y_test):
    model.eval() 
    
    with torch.no_grad(): 
        x_predict=model(x_test) 
        results=criterion(x_predict,y_test) 
    return results.item()

loss2=evaluate(model,criterion,x_test,y_test)
print('평가에 대한 로스(최종 loss):',loss2)
results=model(x_predict.to(DEVICE))
results=results.cpu().detach().numpy()
print('11,12,13 의 예측값:',results)

# 평가에 대한 로스(최종 loss): 0.06450223922729492
# 11,12,13 의 예측값: [[10.8482685]
#  [11.810658 ]
#  [12.773048 ]]