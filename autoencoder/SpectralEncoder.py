import tensorflow as tf
from numpy import ndarray
import time
from tensorflow.examples.tutorials.mnist import input_data

class AutoEncoder:
    def __init__(self, d_visible, d_hidden, x_dtype=None):
        self.d_visible = d_visible
        self.d_hidden = d_hidden

        self.fea = tf.placeholder(dtype=x_dtype, shape=[None, d_visible])

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

    def _create_loss(self):
        output = self.decode(self.fea)
        return tf.squared_difference(output, self.fea)

    def _optimize(self, learn_rate):
        self.loss = self._create_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.loss)

    def train(self, input, learn_rate, no_epochs, btch_size):
        self._optimize(learn_rate)

        if type(input) is ndarray:
            feature = tf.placeholder(dtype=input.dtype, shape=input.shape)
            dataset = tf.data.Dataset.from_tensor_slices(feature).batch(batch_size=btch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            with tf.Session() as sess:
                avg_loss = 0
                t3 = 0
                for i in range(no_epochs):
                    t1 = time.time()
                    sess.run(iterator.initializer, feed_dict={feature: input})
                    try:
                        batch_infer_loss = sess.run([self.loss, self.optimizer, next_batch],
                                                    feed_dict={self.fea: next_batch})
                        avg_loss += batch_infer_loss / input.shape[0]
                    except tf.errors.OutOfRangeError:
                        t2 = time.time() - t1
                        print('epoch {} average loss = {}, time elapsed = {}'.format(i, avg_loss, t2))
                        t3 += t2
                        avg_loss = 0
                        break
                print('total time elapsed = {}'.format(t3))

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(mnist.train.next_batch(5)[0])
    # AE = AutoEncoder()

if __name__ == '__main__':
    main()
