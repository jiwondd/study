import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


datasets=load_breast_cancer()
x,y=datasets.data,datasets.target
y=y.reshape(-1,1)

x_train, x_test, y_train,y_test=train_test_split(x,y,train_size=0.8,
                                                 random_state=123,stratify=y)

# print(x_train.dtype,y_train.dtype) #float64 / int32

print(x_train.shape,y_train.shape) #(455, 30) (455,)

x=tf.compat.v1.placeholder(tf.float32,shape=[None,30])
y=tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1=tf.Variable(tf.zeros([30,10]),name='weight1')
b1=tf.Variable(tf.zeros([10]),name='bias1')
hidden1=tf.compat.v1.matmul(x,w1)+b1

w2=tf.Variable(tf.zeros([10,5]),name='weight2')
b2=tf.Variable(tf.zeros([5]),name='bias2')
hidden2=tf.compat.v1.matmul(hidden1,w2)+b2

w3=tf.Variable(tf.zeros([5,1]),name='weight3')
b3=tf.Variable(tf.zeros([1]),name='bias3')
hypothesis=tf.compat.v1.sigmoid(tf.matmul(hidden2,w3)+b3)

loss=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # =binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.0000001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(1001) : 
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
    feed_dict = { x:x_train, y:y_train})

    if epochs % 100 == 0 :
        print(epochs, "Cost : ",cost_val,"\n",hy_val) 
        
# y_predict=sess.run(tf.cast(hy_val>0.5,dtype=tf.int32))
# acc=accuracy_score(y_data,y_predict)
# print('acc:',acc)
predict=tf.cast(hypothesis>0.5,dtype=tf.float32)
acc=tf.reduce_mean(tf.cast(tf.equal(predict,y),dtype=tf.float32))
c,a=sess.run([predict,acc],feed_dict={x:x_test,y:y_test})
print('acc:',a)

# acc: 0.8947368
# acc: 0.6315789
