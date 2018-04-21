import tensorflow as tf
import numpy as np

a = np.arange(18).reshape(6,3)
b = np.arange(6).reshape(6)
def type1():
    global a, b
    feature = tf.placeholder(dtype=a.dtype, shape=a.shape)
    label = tf.placeholder(dtype=b.dtype, shape=b.shape)
    dataset = tf.data.Dataset.from_tensor_slices({'feature': feature, 'label': label})

    sess = tf.Session()
    dataset = dataset.batch(4)
    dataset = dataset.repeat(3)
    iterator = dataset.make_initializable_iterator()
    next_el = iterator.get_next()
    sess.run(iterator.initializer, feed_dict={feature: a, label: b})
    while True:
        try:
            a = sess.run(next_el)
            print(a)
        except tf.errors.OutOfRangeError:
            break

def type2():
      """"""

if __name__ == '__main__':
    type1()