import tensorflow as tf

sess=tf.compat.v1.Session()

x=tf.Variable([2],dtype=tf.float32)
y=tf.Variable([3],dtype=tf.float32)

init=tf.compat.v1.global_variables_initializer()
sess.run(init) #초기화해서 값을 넣을 상태를 만들어주고 또 실행까지 시켜줘야 돌아간다^^
# 초기화해서 들어갈 자리를 만들어 줘야함...

print(sess.run(x+y))