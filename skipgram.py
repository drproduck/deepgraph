import tensorflow as tf

""" word2vec with NCE loss
and code to visualize the embeddings on TensorBoard
"""

import os
import numpy as np
import tensorflow as tf
import numpy.linalg as la
from tensorflow.contrib.tensorboard.plugins import projector

# from language_models.skipgram.process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
SKIP_STEP = 2000


def feeder(path, mode, window_size):
    f = open(path)
    if mode == 'text':
        word_list = [word for line in f for word in line.split()]
        word_set = set(word_list)
        word_to_idx = dict()
        idx = 0
        for word in word_set:
            word_to_idx[word] = idx
            idx += 1
        idx_list = [word_to_idx[word] for word in word_list]
        while True:
            for i in range(window_size, len(word_list) - window_size):
                for j in range(window_size * 2 + 1):
                    yield ((idx_list[i], idx_list[i - window_size + j]))

    if mode == 'graph':
        g = np.loadtxt(path)
        n = np.size(g, 1)
        m = np.size(g, 2)
        if not n == m: raise Exception('has to be square matrix')
        g = np.divide(g, np.reshape(np.sum(g, 2), [n, 1]))
        while True:
            od = list(range(n))
            np.random.shuffle(od)
            for node in od:
                prev_node = node
                for _ in window_size:
                    draw = np.random.multinomial(1, pvals=g[prev_node])
                    next_node = [i for i in range(n) if draw[i] == 1][0]
                    yield(node, next_node)
                    prev_node = next_node




class SkipGram:
    """ Build the graph for word2vec model """

    def __init__(self, vocab_size, embed_size, batch_size, nsampled, nclass, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size  # size of embed vector
        self.batch_size = batch_size
        self.nsampled = nsampled  # number of contrastive samples (negative samples)
        self.nclass = nclass
        self.lr = learning_rate
        # self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        with tf.device('/cpu:0'):
            with tf.name_scope("embed"):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                                   self.embed_size], -1.0, 1.0),
                                                name='embed_matrix')

    def _create_loss(self):
        """ Step 3 + 4: define the model + the loss function """
        with tf.name_scope("loss"):
            # Step 3: define the inference
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

            # Step 4: define loss function
            # construct variables for NCE loss
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                         stddev=1.0 / (self.embed_size ** 0.5)),
                                     name='nce_weight')
            nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

            # define loss function to be NCE loss function
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                      biases=nce_bias,
                                                      labels=self.target_words,
                                                      inputs=embed,
                                                      num_sampled=self.nsampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        # self._create_summaries()


def train_model(model, batch_gen, num_train_steps):
    saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

    initial_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/skipgram/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
        for index in range(num_train_steps):
            centers, targets = next(batch_gen)
            feed_dict = {model.center_words: centers, model.target_words: targets}
            loss_batch, _, = sess.run([model.loss, model.optimizer],
                                      feed_dict=feed_dict)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/skipgram/skip-gram', index)
