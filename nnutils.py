import tensorflow as tf
import numpy as np
def fclayer(x, w=None, b=None, w_s=None, b_s=None, activation=tf.nn.sigmoid):
    if (w is None and w_s is None) or (b is None and b_s is None):
        raise Exception('Either a variable or shape must be specified.')
    if w is None:
        w = weight(w_s)
    if b is None:
        b = tf.Variable(initial_value=tf.constant(value=0.0, shape=b_s))

    return activation(tf.matmul(x, w)+ b), w, b

def weight(shape):
    """:return weight w that has this shape and random_normal value based on this shape"""
    return tf.Variable(initial_value=tf.random_normal(mean=0.0, stddev=2.0 / np.product(shape)**0.5,
                                                      shape=shape))

def corrupt_input(input, corrupt_level):
    return np.random.binomial(n=1, p=1-corrupt_level, size=input.shape) * input

