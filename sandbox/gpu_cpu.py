import tensorflow as tf

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
with tf.device('/cpu:0'):
    e = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], shape=[2, 9], dtype=tf.float32,
                    name='e')
    f = tf.matmul(c, e)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(f))
