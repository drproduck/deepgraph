from autoencoder.AutoEncoder import AutoEncoder
import nnutils
import tensorflow as tf
import numpy as np

BATCH_SIZE = 100
EPOCHS = 100
def build(d_input, d_output, layers, data):
    x = tf.placeholder(shape=[None, d_input], dtype=tf.float32)
    activation = tf.nn.relu
    wl = []
    bl = []
    tinyaes = []
    for d1, d2 in zip(layers[:-1], layers[1:]):
        ae, w_vis, b_vis = AutoEncoder.build_autoencoder(d1, d2, activation, tied_weight=False, corrupt_level=0.3)
        wl.append(w_vis)
        bl.append(b_vis)
        tinyaes.append(ae)

    y = x
    for i in range(len(wl)):
        y,_,_ = nnutils.fclayer(y, w=wl[i], b=bl[i])

    fine_tune_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(y)
    # dataset = tf.data.Dataset.from_tensor_slices(data)
    # iterator = dataset.batch(BATCH_SIZE).repeat(EPOCHS).make_one_shot_iterator()
    # next_element = iterator.get_next()
    transformed_data = np.copy(data)
    sess = tf.Session()

    #train each autoencoder
    for i in range(len(tinyaes)):
        if not i == len(tinyaes) - 1:
            tinyaes[i].train(transformed_data, batch_size=BATCH_SIZE, no_epochs=EPOCHS, global_session=sess)
            #prepare data for next greedy layer. It is output of previous hidden layer
            transformed_data = tinyaes[i].encode(transformed_data)

    #fine tuning the nn
    fine_tune_op.run(session=sess)

from util import matlabio
fea, gnd = matlabio.getmat('../data/circledata_50', ['fea','gnd'])

d_in = 2
d_out = 1

build(d_in, d_out, layers=[d_in, 10, d_out], data=fea)



