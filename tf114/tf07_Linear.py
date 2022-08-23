# y = wx + b

import tensorflow as tf
tf.set_random_seed(123)
x=[1,2,3]
y=[1,2,3]
w=tf.Variable(11,dtype=tf.float32)
b=tf.Variable(10,dtype=tf.float32)
# ㄴ 웨이트와 바이어스값은 임의로 준거고 연산하면서 갱신된다.

# 2. 모델구성
hypothesis=x*w+b #지금까지는 y=wx+b라고 했는데 사실은 y=xw+b (행렬계산이라고 생각하면 순서가 중요함)

# 3-1. 컴파일
loss=tf.reduce_mean(tf.square(hypothesis-y)) #mse
#                             예측값에서 나온값을 빼고 그거를 제곱(상쇄시켜야하니까) 거리를 구한다. 
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01) #경사하강법...뭐고...
train=optimizer.minimize(loss)
# model.compile(loss='mse',optimizer='sgd') 위에 3줄 앞친게 이거...

# 3-2 훈련
sess=tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step %20==0:
        print(step,sess.run(loss),sess.run(w),sess.run(b))
        
sess.close()



