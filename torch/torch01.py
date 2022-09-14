import numpy as np
import torch
# print(torch.__version__) #1.12.1

import torch.nn as nn #뉴럴네트워크
import torch.optim as optim
import torch.nn.functional as F

# 1.데이터 
x=np.array([1,2,3]) #(3, ) / 스칼라3개 / 벡터1개
y=np.array([1,2,3])

# 텐서형태로 바꿔주기
x=torch.FloatTensor(x).unsqueeze(1) #(3,) -> (3,1)
y=torch.FloatTensor(y).unsqueeze(-1) #맨뒤에 하나의 쉐입을 넣어줄게 / (3,) -> (3,1)
# 언스퀴즈 0 하면 첫번째 자리를 늘려주고 1하면 두번째 자리를 늘려주고 / 리쉐입한다고 생각하면 된다,

print(x,y)
print(x.shape,y.shape)
# tensor([[1.],
#         [2.],
#         [3.]]) tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1]) torch.Size([3, 1])

# 2.모델구성
# model=Sequential() 이거는 텐서플로형태 
model=nn.Linear(1,1) #앞에 1은 (x)인풋, 앞에 1은 (y)아웃풋 / 단층레이어구성

# 3.컴파일 훈련
criterion=nn.MSELoss()
# criterion=표준이라는 뜻 텐서플로였으면 loss=nn.MSELoss() 였을텐데 토치에서는 크리테리온으로 한다.
optimizer=optim.SGD(model.parameters(),lr=0.01) #모델훈련할때 파라미터마다 넣겠다.
# optim.Adam(model.parameters(),lr=0.01) 아담으로 할 수도 있다.
# ㄴ = model.compile(loss='mse',optimizer='SGD')

def train(model,criterion,optimizer,x,y):
    # model.train() #훈련모드로 간다. (생략가능(디폴트)/밑에서 eval이랑 다른점 보여주려고 쓴거임)
    optimizer.zero_grad() #손실함수의 기울기를 초기화한다. / 항상 쓰는 문장이다. 그냥 외우도록 하자
    
    hypothesis=model(x)
    loss=criterion(hypothesis,y)
    
    loss.backward() #역전파 시키겠다.
    optimizer.step() #웨이트를 반환한다. 위 세단계 까지가 1에포 위 3문장은 통으로 외우자!
    return loss.item() #변수값만 가져올게

epochs=1000
for epoch in range(1,epochs+1): # +1안하면 99번 돌겠쥬?
    loss=train(model,criterion,optimizer,x,y)
    print('epoch:{},loss:{}'.format(epoch,loss))
    
    
# 4. 평가, 예측
# loss=model.evaluate(x,y)

def evaluate(model,criterion,x,y): #평가에서는 가중치 갱신을 할 필요가 없으니까 옵티마이저를 안넣어도 됩니다
    model.eval() #평가모드로 간다. *무조건 넣어줘야해*
    
    with torch.no_grad(): #반드시 외우자
        y_predict=model(x) 
        results=criterion(y_predict,y) #여기서도 그라드가 갱신될수 있으니까 확인사살로 위에서 no grad 해주자
    return results.item()

loss2=evaluate(model,criterion,x,y)
print('평가에 대한 로스(최종 loss):',loss2)

results=model(torch.Tensor([[4]]))
# y_predict=model.predict([4])

print('4의 예측값:',results.item())