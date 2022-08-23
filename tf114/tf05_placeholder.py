# 상수, 변수, 플레이스홀더 3가지 자료가 있다.
# 플레이스홀더는 입력에서만 쓴다.

import numpy as np
import tensorflow as tf
print(tf.__version__)

node1=tf.constant(3.0,tf.float32)
node2=tf.constant(4.0)
node3=tf.add(node1,node2)

sess=tf.compat.v1.Session()
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
# 인풋용인데 플레이스홀더가 들어갈 자리를 만들어주면 피드딕에 들어간다,
# 피드딕에는 플레이스홀더가 들어간다. 

add_node=a+b
print(sess.run(add_node,feed_dict={a:3,b:4.5})) # 7.5 
print(sess.run(add_node,feed_dict={a:[1,3],b:[2,4]})) #[3. 7.]

add_and_triple=add_node*3
print(add_and_triple) #Tensor("mul:0", dtype=float32)
print(sess.run(add_and_triple,feed_dict={a:3,b:4.5})) #22.5

