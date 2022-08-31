from tkinter import Y
import numpy as np
import tensorflow as tf
tf.set_random_seed(123)

x_data=[[1,2,1,1],
        [2,1,3,2],
        [3,1,3,4],
        [4,1,5,5],
        [1,7,5,5],
        [1,2,5,6],
        [1,6,6,6],
        [1,7,6,7]]

y_data = [[0,0,1], 
          [0,0,1], 
          [0,0,1],
          [0,1,0], 
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

nb_classes = 3
x=tf.compat.v1.placeholder(tf.float32,shape=[None,4])
w=tf.Variable(tf.random_normal([4,3]),name='weight')
b=tf.Variable(tf.random_normal([1,3]),name='bias')
y=tf.compat.v1.placeholder(tf.float32,shape=[None,3])

hypothesis=tf.nn.softmax(tf.matmul(x,w)+b)
# model.add(Dense(3,activation='softmax',input_dim=4))

loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train=tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) : 
    sess.run(train, feed_dict = {x:x_data, y:y_data})

    if step%2000 == 0 :
        print(step, sess.run([loss,w,b], feed_dict = {x: x_data, y:y_data}))

hypothesis = tf.nn.softmax(tf.matmul(x,w)+b)
a = sess.run(hypothesis, feed_dict = { x:[[1,11,7,9]] })
print(a, sess.run(tf.arg_max(a,1)))
print(a)

