import tensorflow as tf

def layer(x, weight_shape, bias_shape):

    w_init = tf.random_normal_initializer(mean=0.0,
                                          stddev=(2.0/tf.reduce_prod(weight_shape))**0.5)
    b_init = tf.constant_initializer(value=0)

    w = tf.get_variable('w',
                        shape=weight_shape,
                        initializer=w_init)
    b = tf.get_variable('b', shape=bias_shape,
                        initializer=b_init)

    return tf.nn.relu(tf.matmul(x,w), b)

