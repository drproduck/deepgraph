import tensorflow as tf
from numpy import ndarray

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

    def build(self):

    def train(self, input, learn_rate, no_epochs, btch_size):
        if type(input) is ndarray:
            fea = tf.placeholder(dtype=input.dtype, shape=input.shape)
            dataset = tf.data.Dataset.from_tensor_slices(fea).batch(batch_size=btch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            with tf.Session() as sess:
                avg_loss = 0
                for i in range(no_epochs):
                    sess.run(iterator.initializer, feed_dict={fea: input})
                    try:
                        infer_loss = loss(input)



        train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.loss())






