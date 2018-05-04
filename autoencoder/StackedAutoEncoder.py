from autoencoder.AutoEncoder import AutoEncoder
import nnutils
import tensorflow as tf

BATCH_SIZE = 100
EPOCHS = 100
class StackedAE:
    def build(self, d_input, d_output, layers, data):
        x = tf.placeholder(shape=[None, d_input], dtype=tf.float32)
        activation = tf.nn.relu
        wl = []
        bl = []
        optimizers = []
        for d1, d2 in zip(layers[:-1], layers[1:]):
            opt, w_vis, b_vis = AutoEncoder.build_autoencoder(d1, d2, activation, tied_weight=False, corrupt_level=0.3)
            wl.append(w_vis)
            bl.append(b_vis)
            optimizers.append(opt)

        y = x
        for i in range(len(wl)):
            y,_,_ = nnutils.fclayer(y, w=wl[i], b=bl[i])

        fine_tune = tf.train.AdamOptimizer(learning_rate=0.01).minimize(y)
        iterator = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE).repeat(EPOCHS).make_one_shot_iterator()
        next_element = iterator.get_next()

        sess = tf.Session()

        #train each autoencoder
        for i in range(len(wl)):
            optimizers[i](next_element, sess)

        #fine tuning the nn
        fine_tune.run(session=sess)




