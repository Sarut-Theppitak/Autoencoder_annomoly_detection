import tensorflow as tf

a = 2
b = 3
c = tf.add(a, b, name='Add')

sess = tf.Session()
print(sess.run(c))
sess.close()