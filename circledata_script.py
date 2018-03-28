import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import matop
import scipy.sparse as sparse

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
    plt.show()

def main():
    content = scio.loadmat('/home/drproduck/Documents/circledata_50.mat', mat_dtype=True)
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
    # print(np.min(np.max(w, 1), 0))
    # w = np.where(w >= 0.008, 1, 0)
    # print(np.shape(w))
    # print(w[:10, :])

    # take the k maximum values of each row
    k = 3
    lobound = -1
    cols = list()
    row_enum = np.arange(n)
    rows = np.repeat(row_enum[None,:], k, axis=0).reshape(n*k).tolist()
    for i in range(k):
        col_argmax = np.argmax(w, 1)
        w[row_enum, col_argmax] = lobound
        cols.extend(col_argmax)
    sw = sparse.coo_matrix(([1]*(n*k), (rows, cols)), shape=(n,m), dtype=np.float64).tocsr()
    print(sw)
    scio.savemat('circledata_sparse',{'network': sw})

    import utils
    bf = utils.batch_feeder(mat=sw, mode='graph', walk_length=20, window_size=5)
    print(next(bf))
    import matplotlib.pyplot as plt
    import random
    # sample = random.sample(range(2000**2), 10000)
    # plt.hist(w.reshape(2000**2)[sample], bins=100)
    # print(sum(w.reshape(2000**2) >= 0.008) / 4000000)
    # plt.show()
    train(bf, 'circle_embed')
    postprocess('circle_embed.npy', gnd=gnd.reshape(2000))


if __name__ == '__main__':
    main()


