import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수=tf.compat.v1.Variable(tf.random_normal([1]),name='weight')

# 1. 초기화 첫번째
sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa=sess.run(변수)
print('aaa:',aaa) # aaa: [-1.5080816]
sess.close()

# 2. 초기화 두번째
sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb=변수.eval(session=sess) #이 과정을 거쳐야 진정한 변수가 되는거여요
print('bbb:',bbb) # bbb: [-1.5080816]
sess.close()

# 3. 초기화 세번째
sess=tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc=변수.eval()
print('ccc:',ccc)
sess.close()
