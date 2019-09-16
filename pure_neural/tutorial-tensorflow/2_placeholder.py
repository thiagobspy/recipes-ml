import tensorflow as  tf

"""
A placeholder is an object whose value you can specify only later
To specify values for a placeholder, you can pass in values by using a "feed dictionary" (feed_dict variable)
"""

x = tf.placeholder(tf.int64, name='x')
sess = tf.Session()
print(sess.run(2 * x, feed_dict={x: 3}))
