import tensorflow as tf
node1=tf.constant(2.0)
node2=tf.constant(3.0)

# 덧셈
node3=tf.add(node1,node2)
# node3=node1+node2
# 5.0

# 뺄셈
node4=tf.subtract(node2,node1) 
# node4=node2-node1 
#1.0

# 곱셈
node5=tf.multiply(node1,node2)
node5=node1*node2
#6.0

# 나눗셈
node6=tf.div(node2,node1)
# node6=node2/node1
#1.5

'''
실습하기
덧셈 = node3
뺄셈 = node4
곱셈 = node5
나눗셈 = node6
'''

sess=tf.compat.v1.Session()
print(sess.run(node3))