import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import matop

def preprocess():
    content = scio.loadmat('/home/drproduck/Documents/circledata_50.mat', mat_dtype=True)
    global fea, gnd
    fea = content['fea']
    gnd = content['gnd']
    SIGMA = 30

    w = matop.eudist(fea, fea, False)
    n = np.size(w, 0)
    m = np.size(w, 1)
    if not n == m:
        raise Exception('dimensions must agree')
    w = np.exp(-w/(2*(SIGMA**2)))
    w = w - np.eye(n, n)
    print(w[1:10, :])

    import utils
    bf = utils.batch_feeder(mat=w, mode='graph', window_size=20)
    context, target = next(bf)
    print(context, target)
    return bf

def train(bf, save_dir):
    from skipgram import SkipGram

    model = SkipGram(2000, 3, 128, 10, 1.00)
    model.build_graph()
    embedding = model.train_model(bf, 5000, 100)
    print(embedding)
    np.save(save_dir,embedding)

def postprocess(save_dir, gnd=None):
    embedding = np.load(save_dir)
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(111, projection='3d')
    if gnd is not None:
        ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=gnd)
    else:
        ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2])

def main():
    bf = preprocess()
    train(bf, 'circle_embed')
    postprocess('circle_embed', gnd=gnd)


if __name__ == '__main__':
    main()


