from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten #이미지는 2d니까 2d로

model=Sequential()
# model.add(Dense(units=10,input_shape=(3,)))  #input_shape=(10,10,3)))로 바꾸면 아래랑 똑같음
model.add(Conv2D(filters=10,kernel_size=(2,2), #인풋에서 아웃풋하는 노드 수 , kernel_size=크롭 사이즈 
                 input_shape=(5,5,1))) #출력 (4,4,10)
#(batch_size,rows, columns,channels) / 장수N,5,5,1(흑백/컬러일경우 3) 
model.add(Conv2D(7,(2,2),activation='relu')) #출력 (3,3,7)
#               필터=아웃풋노드(커널사이즈) 이미지에서 렐루가 좋은편
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()             
#(kernel_size*chnnerls+bias)*filters = summary param 갯수

# input_dim+bias * units = summary param 갯수 (Dense모델)

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50   -> (2,2)+bias 1 *10
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
_________________________________________________________________
'''

# model=Sequential()
# model.add(Conv2D(filters=10,kernel_size=(4,4), 
#                  input_shape=(7,7,1))) #4,4,10 /170
# model.add(Conv2D(10,(2,2),activation='relu')) #3,3,10 / 410
# model.add(Flatten()) #90
# model.summary()  

# model=Sequential()
# model.add(Conv2D(filters=8,kernel_size=(3,3), 
#                  input_shape=(9,9,1))) # 7,7,8 / 80
# model.add(Conv2D(8,(4,4),activation='relu')) #4,4,8 / 1032
# model.add(Flatten()) #128
# model.summary()  

# model=Sequential()
# model.add(Conv2D(filters=12,kernel_size=(4,4), 
#                  input_shape=(8,8,1))) #(8-4+1=)5,5,12 /(4*4+1*12=)204  
# model.add(Conv2D(10,(2,2),activation='relu')) # (5-2+1=)4,4,10 / (2*2*12+1*10=)490
# model.add(Flatten()) #(4*4**10=)160
# model.summary()  

