# LR=Learning Rate수정해서 epochs를 100이하로 줄인다.
# step=100이하, w=1.99, b=0.99  


import tensorflow as tf
tf.set_random_seed(123)

# 1.데이터
x_train_data=[1,2,3]
y_train_data=[3,5,7]
x_test_data=[6,7,8]
x_train=tf.placeholder(tf.float32,shape=[None])
y_train=tf.placeholder(tf.float32,shape=[None])
x_test=tf.compat.v1.placeholder(tf.float32,shape=[None])
w=tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b=tf.Variable(tf.random_normal([1]),dtype=tf.float32)

# 2. 모델구성
hypothesis=x_train*w+b #지금까지는 y=wx+b라고 했는데 사실은 y=xw+b (행렬계산이라고 생각하면 순서가 중요함)

# 3-1. 컴파일
loss=tf.reduce_mean(tf.square(hypothesis-y_train))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.175)
train=optimizer.minimize(loss)

# 3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        # sess.run(train)
        _, loss_val, w_val,b_val=sess.run([train,loss,w,b],
                 feed_dict={x_train:x_train_data,y_train:y_train_data})

        if step %10==0:
            # print(step,sess.run(loss),sess.run(w),sess.run(b))
            print(step,loss_val,w_val,b_val)
            
    y_predict=x_test*w_val+b_val
    print('[6,7,8]예측:',sess.run(y_predict,feed_dict={x_test:x_test_data}))
     
# 100 0.087880254 [2.3434644] [0.21919316]
# [6,7,8]예측: [14.27998  16.623444 18.966908] 

# 100 0.0010949888 [2.0375087] [0.91473377]
# [6,7,8]예측: [13.139786 15.177295 17.214804 0.1

# 100 9.1841095e-05 [2.010729] [0.97561026]
# [6,7,8]예측: [13.039986  15.0507145 17.061443 ] 0.15

# 100 0.00036618105 [2.0124238] [0.9901826]
# [6,7,8]예측: [13.064726 15.077149 17.089573] 0.1755

# 100 0.00013141616 [2.0094557] [0.9886817]
# [6,7,8]예측: [13.045416 15.054872 17.064327] 0.175


