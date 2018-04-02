import numpy as np
from time import time
import scipy.sparse as sp
import scipy
import scipy.io as scio
from sklearn.utils.extmath import randomized_svd
from spectralclustering import spectralclustering as sc
import utils.io as process

def svd_speed_test(mat):

    print('using numpy svd')
    t = time()
    np.svd(mat, full_matrices=False, compute_uv=True)
    print('time elapsed for numpy svd: {}'.format(time() - t))

    print('using scipy sparse svds')
    t = time()
    sp.svds(mat, 10)
    print('time elapsed for scipy svds: {}'.format(time() - t))

    print('using scipy svd')
    t = time()
    scipysvd(mat ,full_matrices=False, compute_uv=True)
    print('time elapsed for scipy svd: {}'.format(time() - t))

    print('using scikit-learn randomized svd')
    t = time()
    randomized_svd(mat, n_components=10)
    print('time elapsed for scikit-learn randomized svd: {}'.format(time() - t))
    return time

def test_plusplus(path):
    from time import time
    content = scipy.io.loadmat(path, mat_dtype=True)
    fea = content['fea']
    print(fea.shape)
    # gnd = content['gnd'].reshape(2000)
    a = time()
    _, idx1 = sc.pick_representatives(fea, 1000, mode='random')
    b = time()
    print('time elapsed for random: {}'.format(b - a))
    a = time()
    _, idx2 = sc.pick_representatives(fea, 1000, mode='++')
    b = time()
    print('time elapsed for ++: {}'.format(b - a))
    # import matplotlib.pyplot as plt
    # color = np.zeros(shape=2000, dtype=np.int32)
    # color[idx1] = 1
    # plt.figure(0)
    # plt.scatter(fea[:,0], fea[:,1], c=color)
    # color = np.zeros(shape=2000, dtype=np.int32)
    # color[idx2] = 1
    # plt.figure(1)
    # plt.scatter(fea[:,0], fea[:,1], c=color)

    # plt.show()

def test_spectral_clustering(path):
    content = scipy.io.loadmat('../data/circledata_50.mat', mat_dtype=True)
    fea = content['fea']
    gnd = content['gnd'].reshape(2000)
    import matplotlib.pyplot as plt
    labels = sc.spectral_clustering(fea, 2, affinity='gaussian', sigma=30)
    labels = [int(x) for x in labels]
    gnd = [int(x) for x in gnd]
    # print(np.concatenate((labels, gnd)))
    labels, diff = process.greedy_matching(labels, gnd)
    plt.scatter(fea[:,0], fea[:,1], c=labels)
    plt.show()

def test_bipartite_clustering(path):
    content = scio.loadmat(path, mat_dtype=True)
    fea = content['fea']
    gnd = content['gnd']
    gnd = gnd.reshape(gnd.size).astype(np.int64)
    n_label = int(np.max(gnd))
    import matplotlib.pyplot as plt
    labels = sc.bipartite_clustering(fea, n_label, 'gaussian', n_reps=500, select_method='++',
                                  sparsity=3, use_embedding='v', sigma=1)
    print(labels)
    print(gnd)
    labels, diff = process.best_map(labels, gnd)
    # plt.scatter(fea[:,0], fea[:,1], c=labels)
    print(labels)
    print(gnd)
    print(process.accuracy(labels, gnd))
    # plt.show()
def main():
    # test_plusplus('../data/news.mat')
    # test_spectral_clustering('../data/circledata_50.mat')
    # mat = np.random.randn(2000, 2000)
    # svd_speed_test(mat)
    test_bipartite_clustering('../data/mnist.mat')


if __name__ == '__main__':
    main()
