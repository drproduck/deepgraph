import tensorflow as tf
from numpy import ndarray
import time
from tensorflow.examples.tutorials.mnist import input_data

class AutoEncoder:
    def __init__(self, d_visible, d_hidden, x_dtype=None):
        self.d_visible = d_visible
        self.d_hidden = d_hidden
        self.fea = tf.placeholder(dtype=x_dtype, shape=[None, d_visible])
        self.learn_rate = tf.placeholder(dtype=tf.float32)
    def _make_encoder(self):
        with tf.name_scope('encode'):
            w = tf.Variable(name='w',
                            initial_value=tf.random_normal(shape=[self.d_visible, self.d_hidden],
                                             mean=0.0,
                                             stddev=(2.0/(self.d_visible+self.d_hidden))**0.5)
                            )
            b = tf.Variable(name='b',
                            initial_value=tf.constant(value=0.0, shape=[self.d_hidden])
                            )
            self.encode = tf.nn.relu(tf.matmul(self.fea,w)+b)

    def code(self,x):
        return self.sess.run(self.encode, feed_dict={self.fea: x})

    def _make_decoder(self):
        with tf.name_scope('decode'):
            w = tf.Variable(name='w',
                            initial_value=tf.random_normal(shape=[self.d_hidden, self.d_visible],
                                             mean=0.0,
                                             stddev=(2.0/(self.d_visible+self.d_hidden))**0.5)
                            )
            b = tf.Variable(name='b',
                            initial_value=tf.constant(value=0.0,shape=[self.d_visible])
                            )
            self.decode = tf.nn.relu(tf.matmul(self.encode,w)+b)

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.fea, self.decode))

    def _optimize(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

    def build(self):
        self._make_encoder()
        self._make_decoder()
        self._create_loss()
        self._optimize()

    def train(self, input, learn_rate, no_epochs, batch_size):

        if type(input) is ndarray:
            feature = tf.placeholder(dtype=input.dtype, shape=input.shape)
            dataset = tf.data.Dataset.from_tensor_slices(feature).batch(batch_size=batch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()

            self.sess = tf.Session()
            avg_loss = 0
            t3 = 0
            self.sess.run(tf.global_variables_initializer())
            for i in range(no_epochs):
                t1 = time.time()
                self.sess.run(iterator.initializer, feed_dict={feature: input})
                while True:
                    try:
                        batch =  self.sess.run(next_batch)
                        batch_infer_loss,_ = self.sess.run([self.loss, self.optimizer],
                                                    feed_dict={self.fea: batch,
                                                               self.learn_rate: learn_rate})
                        avg_loss += batch_infer_loss / input.shape[0]
                    except tf.errors.OutOfRangeError:
                        t2 = time.time() - t1
                        print('epoch {} average loss = {}, time elapsed = {}'.format(i, avg_loss, t2))
                        t3 += t2
                        avg_loss = 0
                        break
            print('total time elapsed = {}'.format(t3))
            print(self.sess.run(self.encode, feed_dict={self.fea: input}))

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    input = mnist.train.next_batch(10000)[0]
    print(input.shape)
    AE = AutoEncoder(d_visible=784, d_hidden=2, x_dtype=tf.float32)
    AE.build()
    AE.train(input, learn_rate=1.0, no_epochs=10, batch_size=128)
    import matplotlib.pyplot as plt
    points = AE.code(input)
    print(points.shape)
    plt.scatter(points[:,0], points[:,1])
    plt.show()

if __name__ == '__main__':
    main()
