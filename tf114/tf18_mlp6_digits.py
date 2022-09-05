import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)
from sklearn.datasets import load_iris,load_wine,load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd

datasets = load_digits()
x_data = datasets.data      
y_data = datasets.target     
y_data = pd.get_dummies(y_data)
# y_data =y_data.reshape(-1, 1)
# print(y_data)

print(x_data.shape,y_data.shape)  # (1797, 64) (1797, 10)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=1234)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,64])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,10])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,10]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,10]))
hidden=tf.compat.v1.matmul(x,w1)+b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,5]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,5]))
hidden2=tf.compat.v1.matmul(hidden,w2)+b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([5,1]))
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,1]))
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden2,w3)+b3)

loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
train = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 1500
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   
y_predict = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})

y_predict = np.argmax(y_predict,axis=1)
y_test = y_test.values
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_predict)
print('acc :',acc)

# acc : 0.07777777777777778
# acc : 0.07777777777777778
