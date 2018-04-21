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
    dataset = dataset.repeat(3) # only use repeat if not also use OutOfRange signal
    # iterator = dataset.make_initializable_iterator()
    iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)
    next_el = iterator.get_next()

    #go through dataset once, 1 epoch
    # sess.run(iterator.initializer, feed_dict={feature: a, label: b})
    # while True:
    #     try:
    #         a = sess.run(next_el)
    #         print(a)
    #     except tf.errors.OutOfRangeError:
    #         break

    #multiple epochs
    for _ in range(2):
        train_iter_op = iterator.make_initializer(dataset)
        sess.run(train_iter_op, feed_dict={feature: a, label: b})
        while True:
            try:
                c = sess.run(next_el)
                print(c)
            except tf.errors.OutOfRangeError:
                break

def type2():
    global a,b
    feature = tf.placeholder(dtype=a.dtype, shape=a.shape)
    label = tf.placeholder(dtype=b.dtype, shape=b.shape)
    dataset = tf.data.Dataset.from_tensor_slices({'feature': feature, 'label': label})
    sess = tf.Session()
    dataset = dataset.batch(3)

    iterator = dataset.make_initializable_iterator()
    next_el = iterator.get_next()

    for _ in range(2):
        sess.run(iterator.initializer, feed_dict={feature: a, label: b})
        while True:
            try:
                c = sess.run(next_el)
                print(c)
            except tf.errors.OutOfRangeError:
                break

def type3():
    global a,b
    feature = tf.placeholder(dtype=a.dtype, shape=a.shape)
    label = tf.placeholder(dtype=b.dtype, shape=b.shape)
    dataset = tf.data.Dataset.from_tensor_slices({'feature': feature, 'label': label})
    dataset = dataset.batch(3)
    no_epochs = 3
    dataset = dataset.repeat(no_epochs)
    iterator = dataset.make_initializable_iterator()
    next_el = iterator.get_next()

    with tf.train.MonitoredTrainingSession() as sess:
        sess.run(iterator.initializer, feed_dict={feature: a, label: b})
        while not sess.should_stop():
            c = sess.run(next_el)
            print(c['feature'], c['label'])

if __name__ == '__main__':
    type3()