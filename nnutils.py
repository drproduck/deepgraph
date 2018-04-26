import tensorflow as tf

def fclayer(x, w_s, b_s, variable_scope=None, activation=tf.nn.sigmoid):
    if variable_scope is not None:
        with tf.variable_scope(variable_scope):
            w_init = tf.random_normal_initializer(mean=0.0,
                                                  stddev=(2.0 / tf.reduce_prod(w_s)) ** 0.5)
            b_init = tf.constant_initializer(value=0)

            w1 = tf.get_variable('w',
                                 shape=w_s,
                                 initializer=w_init)
            b1 = tf.get_variable('b', shape=b_s,
                                 initializer=b_init)

            return activation(tf.matmul(x, w1), b1)
    elif variable_scope is None:
        w = weight(w_s)
        b = tf.Variable(initial_value=tf.constant(value=0.0, shape=b_s))
        return w, b, activation(tf.matmul(x, w), b)

def weight(shape):
    """:return weight w that has this shape and random_normal value based on this shape"""
    return tf.Variable(initial_value=tf.random_normal(mean=0.0, stddev=2.0 / (shape[0] * shape[1])**0.5))
