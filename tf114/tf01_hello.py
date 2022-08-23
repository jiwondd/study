import tensorflow as tf

print(tf.__version__)

# print('hello world')
hello=tf.constant('hello world')
# tf는 변수, 상수, 하나 더 있음
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess=tf.Session()
# WARNING:tensorflow:From c:/study/tf114/tf01_hello.py:9: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
sess=tf.compat.v1.Session()
print(sess.run(hello))

# 텐서플로1은 반드시 출력을 할때 sess run을 거쳐야 한다.

 