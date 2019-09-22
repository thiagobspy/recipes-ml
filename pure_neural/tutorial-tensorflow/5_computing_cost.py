import tensorflow as tf


def cost(logits, labels):
    z = tf.placeholder(tf.float32, name='logits')
    y = tf.placeholder(tf.float32, name='labels')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    sess = tf.Session()

    cost = sess.run(cost, feed_dict={z: logits, y: labels})

    sess.close()

    return cost
