import autoencoder.AutoEncoder as ae
import tensorflow as tf

tinyae, w, b = ae.AutoEncoder.build_autoencoder(d_hidden=1, d_visible=2, activation=tf.nn.sigmoid, tied_weight=True)
import scipy.io as sio

content = sio.loadmat('../data/circledata_50.mat', mat_dtype=True)
fea = content['fea']
sess = tf.Session()
tinyae.train(fea, learn_rate=0.01, no_epochs=10, batch_size=20, global_session=sess, corrupt_level=0.3)