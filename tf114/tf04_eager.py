import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) #eagerly=열심히
# 1.14.0
# False
# 즉시실행모드!!
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

hello=tf.constant('hello world')
sess=tf.compat.v1.Session()
print(sess.run(hello))

# 즉시실행모드는? 2.0 대의 버전이고 (true)
# 1.0대의 모드는 즉시실행모드가 아니다. 그니까 텐서2에서 텐서1처럼 쓰고싶으면
# tf.compat.v1.disable_eager_execution() <-이거 넣어주면 된다.
