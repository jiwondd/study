import tensorflow as tf
tf.set_random_seed(123)


x=tf.placeholder(tf.float32,shape=[None])
y=tf.placeholder(tf.float32,shape=[None])

w=tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b=tf.Variable(tf.random_normal([1]),dtype=tf.float32)

# 2. 모델구성
hypothesis=x*w+b #지금까지는 y=wx+b라고 했는데 사실은 y=xw+b (행렬계산이라고 생각하면 순서가 중요함)

# 3-1. 컴파일
loss=tf.reduce_mean(tf.square(hypothesis-y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss)

# 3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # sess.run(train)
        _, loss_val, w_val,b_val=sess.run([train,loss,w,b],
                 feed_dict={x:[1,2,3,4,5],y:[1,2,3,4,5]})
        if step %20==0:
            # print(step,sess.run(loss),sess.run(w),sess.run(b))
            print(step,loss_val,w_val,b_val)
        
# 언더바는 반환은 해주지 않겠지만 실행을 하겠다. 

