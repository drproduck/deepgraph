import tensorflow as tf

class AutoEncoder:
    def __init__(self, d_visible, d_hidden, x=None):
        self.d_visible = d_visible
        self.d_hidden = d_hidden
        self.x = x

    def encode(self,input):
        with tf.name_scope('encode'):
            w = tf.variable(tf.random_normal(shape=[self.d_visible, self.d_hidden], mean=0.0, no_sttdev=(2.0/(self.d_visible+self.d_hidden))**0.5))
            b = tf.variable(tf.constant(value=0.0, shape=[self.d_hidden]))
        return tf.sigmoid(tf.matmul(input,w), b)

    def decode(self,input):
        with tf.name_scope('decode'):
            w = tf.variable(tf.random_normal(shape=[self.d_hidden, self.d_visible],
                                             mean=0.0,
                                             stddev=(2.0/(self.d_visible+self.d_hidden))**0.5))
            b = tf.variable(tf.constant(value=0.0,shape=[self.d_visible]))
        return tf.sigmoid(tf.matmul(input,w), b)
    def loss(self,input):
        output = self.decode(self.encode(input))
        return tf.squared_difference(output, input)

    def train(self, learn_rate):
        for i in range(num_batch):

        train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.loss())






