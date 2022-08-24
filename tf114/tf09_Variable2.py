import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# 8-2을 카피해서 아래를 만들어보자
# session() / 변수.eval(session=sess)
# InteractiveSession() / 변수.eval()

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

# 3. 컴파일
loss=tf.reduce_mean(tf.square(hypothesis-y_train))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.175)
train=optimizer.minimize(loss)

# 4-1 session() / sess.run(변수)
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

#######################################################################################
# 4-2 session() / 변수.eval(session=sess)
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
    print('[6,7,8]예측:',y_predict.eval(session=sess,feed_dict={x_test:x_test_data}))

#######################################################################################

# 4-3 InteractiveSession() / 변수.eval()
sess=tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())

for step in range(101):
        # sess.run(train)
    _, loss_val, w_val,b_val=sess.run([train,loss,w,b],
                 feed_dict={x_train:x_train_data,y_train:y_train_data})

    if step %10==0:
            # print(step,sess.run(loss),sess.run(w),sess.run(b))
            print(step,loss_val,w_val,b_val)
            
y_predict=x_test*w_val+b_val
print('[6,7,8]예측:',y_predict.eval(feed_dict={x_test:x_test_data}))

sess.close()