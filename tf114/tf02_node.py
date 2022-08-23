import tensorflow as tf

node1=tf.constant(3.0,tf.float32)
node2=tf.constant(4.0)
# node3=node1+node2
node3=tf.add(node1,node2)
# print(node3)
# Tensor("add:0", shape=(), dtype=float32)

# sess=tf.Session() 버전 문제로 워닝이 뜰 수 있다 그럴땐 밑에 v1을 사용하면 된다.
sess=tf.compat.v1.Session()
print(sess.run(node3)) #7.0

