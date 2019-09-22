import tensorflow as tf


def ones(shape):
    ones = tf.ones(shape)

    sess = tf.Session()

    ones = sess.run(ones)

    sess.close()

    return ones


def zeros(shape):
    zeros = tf.zeros(shape)

    sess = tf.Session()

    zeros = sess.run(zeros)

    sess.close()

    return zeros
