import tensorflow as tf
tf.set_random_seed(123)
x=[1,2,3,4,5]
y=[1,2,3,4,5]
w=tf.Variable(11,dtype=tf.float32)
b=tf.Variable(10,dtype=tf.float32)

# 2. 모델구성
hypothesis=x*w+b #지금까지는 y=wx+b라고 했는데 사실은 y=xw+b (행렬계산이라고 생각하면 순서가 중요함)

# 3-1. 컴파일
loss=tf.reduce_mean(tf.square(hypothesis-y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss)

# 3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1500):
        sess.run(train)
        if step %20==0:
            print(step,sess.run(loss),sess.run(w),sess.run(b))
        
# sess.close()

#클로즈 하지 않고 위드문으로 사용가능


