import tensorflow as tf


def one_hot_matrix(labels, C):
    C = tf.constant(C)

    one_hot_matrix = tf.one_hot(labels, C, axis=1)

    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)

    sess.close()

    return one_hot
