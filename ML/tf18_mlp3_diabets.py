import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets=load_diabetes()
x_data,y_data=datasets.data,datasets.target

print(x_data.shape,y_data.shape) #(442, 10) (442,)
y_data=y_data.reshape(442,1)

x=tf.compat.v1.placeholder(tf.float32,shape=[None,10])
y=tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal([10,10]),name='weight')
b1=tf.compat.v1.Variable(tf.compat.v1.random_normal([10]),name='bias')
hidden1=tf.compat.v1.matmul(x,w1)+b1

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal([10,5]),name='weight')
b2=tf.compat.v1.Variable(tf.compat.v1.random_normal([5]),name='bias')
hidden2=tf.compat.v1.matmul(hidden1,w2)+b2

w3=tf.compat.v1.Variable(tf.compat.v1.random_normal([5,1]),name='weight')
b3=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias')
hypothesis=tf.compat.v1.matmul(hidden2,w3)+b3

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001) : 
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
    feed_dict = { x:x_data, y : y_data})

    if epochs % 100 == 0 :
        print(epochs, "Cost : ",cost_val,"\n",hy_val) 
        
from sklearn.metrics import r2_score,mean_absolute_error
r2=r2_score(y_data,hy_val)
print('r2:',r2)
mae=mean_absolute_error(y_data,hy_val)
print('mae:',mae)

# r2: -3.601226086685707
# mae: 146.16821460918064

# r2: 0.18163796886133532
# mae: 59.68101781094236