import tensorflow as tf
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score
tf.compat.v1.set_random_seed(123)

# 1. 데이터 
x_data=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] #(6,2)
y_data=[[0],[0],[0],[1],[1],[1]] #(6,1)

x=tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y=tf.compat.v1.placeholder(tf.float32,shape=[None,1])
w=tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')
b=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias')

hypothesis=tf.compat.v1.sigmoid(tf.matmul(x,w)+b)
# model.add(Dense(1,activation='sigmoid',input_dim=2))

# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # =binary_crossentropy
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

y_predict=sess.run(tf.cast(hy_val>0.5,dtype=tf.float32))

r2=r2_score(y_data,hy_val)
print('r2:',r2)
acc=accuracy_score(y_data,y_predict)
print('acc:',acc)
mae=mean_absolute_error(y_data,hy_val)
print('mae:',mae)

# r2: -0.9995880000283626
# acc: 0.5
# mae: 0.5016512904912815