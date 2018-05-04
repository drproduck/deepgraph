import tensorflow as tf
from numpy import ndarray
import time
import utils.io
import spectralclustering.spectralclustering as sc
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from magenta.models.image_stylization.image_utils import form_image_grid
import nnutils as nn

class AutoEncoder:
    def __init__(self, d_visible, d_hidden):
        self.d_visible = d_visible
        self.d_hidden = d_hidden
        self.fea = tf.placeholder(dtype=tf.float32, shape=[None, d_visible])
        self.learn_rate = tf.placeholder(dtype=tf.float32)
        self.encoder = None
        self.decoder = None
        self.loss = None
        self.optimizer = None

    @staticmethod
    def build_autoencoder(d_hidden, d_visible, activation=tf.nn.sigmoid, tied_weight=True, corrupt_level=None):
        """build a basic auto encoder with 1 hidden layer. Good for training multiple stacked autoencoders
        :return an instance of AutoEncoder"""
        ae = AutoEncoder(d_visible, d_hidden)

        ae.fea = tf.placeholder(shape=[None,d_visible],dtype=tf.float32)
        ae.encoder, w_vis, b_vis = nn.fclayer(x=ae.fea, w_s=[d_visible, d_hidden], b_s=[d_hidden], activation=activation)
        if tied_weight:
            b_h = tf.Variable(tf.constant(value=0.0, shape=[d_visible]))
            ae.decoder = activation(tf.matmul(ae.encoder, w_vis, transpose_b=True) + b_h)
            ae.loss = tf.reduce_mean(tf.squared_difference(ae.decoder, ae.fea))
            ae.optimizer = tf.train.AdamOptimizer(0.01).minimize(ae.loss, var_list=[w_vis,b_vis,b_h])
        else:
            ae.decoder, w_h, b_h  = nn.fclayer(x=ae.encoder, w_s=[d_hidden,d_visible], b_s=[d_visible],activation=activation)
            ae.loss = tf.reduce_mean(tf.squared_difference(ae.decoder, ae.fea))
            ae.optimizer = tf.train.AdamOptimizer(0.01).minimize(ae.loss, var_list=[w_vis,b_vis,w_h,b_h])

        return ae, w_vis, b_vis

    def _make_encoder(self, layerdims=None):
        if layerdims is None:
            layerdims = [self.d_visible, self.d_hidden]
        else: layerdims = [self.d_visible] + layerdims + [self.d_hidden]
        with tf.name_scope('encode'):
            self.encoder = self.fea
            for i in range(len(layerdims) - 1):
                dim1 = layerdims[i]; dim2 = layerdims[i+1]
                w = tf.Variable(name='w'+str(i),
                                initial_value=tf.random_normal(shape=[dim1, dim2],
                                                 mean=0.0,
                                                 stddev=(2.0/(dim1 + dim2))**0.5)
                                )
                b = tf.Variable(name='b'+str(i),
                                initial_value=tf.constant(value=0.0, shape=[dim2])
                                )
                self.encoder = tf.nn.relu(tf.matmul(self.encoder, w) + b)

    def encode(self,x):
        return self.sess.run(self.encoder, feed_dict={self.fea: x})

    def _make_decoder(self, layerdims=None):
        if layerdims is None:
            layerdims = [self.d_hidden, self.d_visible]
        else: layerdims = [self.d_hidden] + layerdims + [self.d_visible]
        with tf.name_scope('decode'):
            self.decoder = self.encoder
            for i in range(len(layerdims) - 1):
                dim1 = layerdims[i]; dim2 = layerdims[i+1]
                w = tf.Variable(name='w'+str(i),
                                    initial_value=tf.random_normal(shape=[dim1, dim2],
                                                 mean=0.0,
                                                 stddev=(2.0/(dim1 + dim2))**0.5)
                                )
                b = tf.Variable(name='b'+str(i),
                                initial_value=tf.constant(value=0.0,shape=[dim2])
                                )
                self.decoder = tf.nn.relu(tf.matmul(self.decoder, w) + b)

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.fea, self.decoder))

    def _optimize(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)


    def _make_summaries(self):
        writer = tf.summary.FileWriter("./logs")
        tf.summary.scalar("Loss", self.loss)
        layer_grid_summary("Input", self.fea, [28, 28])
        layer_grid_summary("Encoder", self.encoder, [2, 1])
        layer_grid_summary("Output", self.decoder, [28, 28])
        return writer, tf.summary.merge_all()

    def build(self, encoder_dims, decoder_dims, summarize=False):
        self._make_encoder(encoder_dims)
        self._make_decoder(decoder_dims)
        self._create_loss()
        self._optimize()
        if summarize:
            self._make_summaries()

    def build_simple(self, summarize=False):
        self._make_encoder()
        self._make_decoder()
        self._create_loss()
        self._optimize()
        if summarize:
            self._make_summaries()

    def train(self, input, learn_rate, no_epochs, batch_size, global_session=None, corrupt_level=None):

        assert type(input) is ndarray
        feature = tf.placeholder(dtype=input.dtype, shape=input.shape)
        dataset = tf.data.Dataset.from_tensor_slices(feature).batch(batch_size=batch_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        # corrupt input
        if corrupt_level is not None:
            next_batch = nn.corrupt_input(next_batch, corrupt_level=corrupt_level)

        if global_session is None:
            sess = tf.Session()
        else: sess = global_session
        avg_loss = 0
        t3 = 0
        sess.run(tf.global_variables_initializer())
        for i in range(no_epochs):
            t1 = time.time()
            sess.run(iterator.initializer, feed_dict={feature: input})
            while True:
                try:
                    batch =  sess.run(next_batch)
                    batch_infer_loss,_ = sess.run([self.loss, self.optimizer],
                                                feed_dict={self.fea: batch,
                                                           self.learn_rate: learn_rate})
                    avg_loss += batch_infer_loss / input.shape[0]
                except tf.errors.OutOfRangeError:
                    t2 = time.time() - t1
                    print('epoch {}, average loss = {}, time elapsed = {}'.format(i, avg_loss, t2))
                    t3 += t2
                    avg_loss = 0
                    break
        print('total time elapsed = {}'.format(t3))
        print(sess.run(self.encoder, feed_dict={self.fea: input}))



    # some test functions
    def train_mnist(self, learn_rate, no_epochs, batch_size):
        mnist = input_data.read_data_sets('MNIST_data')
        writer, summary_op = self._make_summaries()
        first_batch = mnist.train.next_batch(batch_size)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(make_image("images/input.jpg", self.fea, [28, 28]), feed_dict={self.fea :
                                                                             first_batch[0]})
        for i in range(no_epochs):
            batch = mnist.train.next_batch(batch_size)
            if i % 500 == 0:
                summary, train_loss = sess.run([summary_op, self.loss],
                                               feed_dict={self.fea: batch[0]})
                print("step %d, training loss: %g" % (i, train_loss))

                writer.add_summary(summary, i)
                writer.flush()

            if i % 1000 == 0:
                sess.run(make_image("images/output_%06i.jpg" % i, self.decoder, [28,
                                                                                 28]), feed_dict={self.fea : first_batch[0]})
            sess.run(self.optimizer, feed_dict={self.fea: batch[0], self.learn_rate: learn_rate})

        # Save latent space
        pred = sess.run(self.encoder, feed_dict={self.fea : mnist.test._images})
        pred = np.asarray(pred)
        pred = np.reshape(pred, (mnist.test._num_examples, 2))
        labels = np.reshape(mnist.test._labels, (mnist.test._num_examples, 1))
        pred = np.hstack((pred, labels))

        np.savetxt("latent.csv", pred)

    def close_sess(self):
        self.sess.close()

class SpectralAE(AutoEncoder):

    @staticmethod
    def make_normalized_affinity(self, input, sigma):
        w = utils.io.make_weight_matrix(input, 'gaussian', sigma=sigma)
        dwd = sc.symmetric_laplacian(w)
        return dwd

    def _make_decoder(self, layerdims):
        """decoder reverse order of encoder's weights"""
        if layerdims is None:
            layerdims = [self.d_visible, self.d_hidden]
        else: layerdims = [self.d_visible] + layerdims + [self.d_hidden]
        self.decode = self.encoder
        for i in reversed(range(len(layerdims) - 1)):

            #tied weight with encoder
            with tf.name_scope('encode'):
                w = tf.get_variable('w'+str(i))
            with tf.name_scope('decode'):
                b = tf.Variable(name='b'+str(i),
                                initial_value=tf.constant(value=0.0,shape=[layerdims[i]])
                                )
                self.decode = tf.tanh(tf.matmul(self.decode, w, transpose_b=True) + b)

    def build(self, layerdims):
        self._make_encoder(layerdims)
        self._make_decoder(layerdims)
        self._create_loss()
        self._optimize()
        self._make_summaries()

    def train(self,input, learn_rate, no_epochs, batch_size, normalized_weight=True, sigma=None):

        if normalized_weight:
            super().train(input, learn_rate,no_epochs, batch_size)
        else:
            normed_input = self.make_normalized_affinity(input, sigma)
            super().train(normed_input, learn_rate,no_epochs,batch_size)


def test_autoencoder_circledata():
    import scipy.io
    content = scipy.io.loadmat('../data/circledata_50.mat', mat_dtype=True)
    no_sample = 1000
    perm = np.random.choice(2000, no_sample, replace=False)
    input = content['fea']
    output = content['gnd']
    input = input[perm,:]
    output = output[perm,:]
    print(input.shape)
    AE = AutoEncoder(d_visible=2, d_hidden=1, x_dtype=tf.float32)
    AE.build()
    AE.train(input, learn_rate=0.01, no_epochs=1000, batch_size=200)
    import matplotlib.pyplot as plt
    points = AE.code(input)
    print(points.dtype)
    points[points < 0.5] = -1
    points[points >= 0.5] = 1
    plt.scatter(input[:,0], input[:,1], c=points.reshape(no_sample), marker='.')
    plt.show()

GRID_ROWS = 5
GRID_COLS = 10
BATCH_SIZE = 50

def layer_grid_summary(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS,
                                                                GRID_COLS], image_dims, 1)
    return tf.summary.image(name, grid)

def make_image(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS,
                                                                 GRID_COLS], image_dims, 1)
    s_grid = tf.squeeze(grid, axis=0)

    # This reproduces the code in: tensorflow/core/kernels/summary_image_op.cc
    im_min = tf.reduce_min(s_grid)
    im_max = tf.reduce_max(s_grid)

    kZeroThreshold = tf.constant(1e-6)
    max_val = tf.maximum(tf.abs(im_min), tf.abs(im_max))

    offset = tf.cond(
        im_min < tf.constant(0.0),
        lambda: tf.constant(128.0),
        lambda: tf.constant(0.0)
    )
    scale = tf.cond(
        im_min < tf.constant(0.0),
        lambda: tf.cond(
            max_val < kZeroThreshold,
            lambda: tf.constant(0.0),
            lambda: tf.div(127.0, max_val)
        ),
        lambda: tf.cond(
            im_max < kZeroThreshold,
            lambda: tf.constant(0.0),
            lambda: tf.div(255.0, im_max)
        )
    )
    s_grid = tf.cast(tf.add(tf.multiply(s_grid, scale), offset), tf.uint8)
    enc = tf.image.encode_jpeg(s_grid)

    fwrite = tf.write_file(name, enc)
    return fwrite


def test_autoencoder_mnist():
    AE = AutoEncoder(d_visible=784, d_hidden=2, x_dtype=tf.float32)
    AE.build(encoder_dims=[50,50], decoder_dims=[50,50])
    AE.train_mnist(learn_rate=1e-2, no_epochs=50000, batch_size=BATCH_SIZE)

def test_spectralencoder():
    """"""

def main():
    test_autoencoder_mnist()

if __name__ == '__main__':
    main()
