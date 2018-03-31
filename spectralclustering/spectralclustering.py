import numpy as np
from numpy.random import choice
from numpy.linalg import svd as numpysvd
import scipy.io
from scipy.sparse.linalg import svds
from scipy.linalg.decomp_svd import svd as scipysvd
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans, k_means_
from time import time
from utils.io import make_weight_matrix, greedy_matching
from utils.matop import eudist, cumdist_matrix
from random import sample
import scipy.sparse as sp

def _normalize_svd(fea, k, mode, sigma=None, normalize=True):
    if mode == 'gaussian' and sigma is None:
        raise Exception('mode Gaussian requires sigma specified')
    w = make_weight_matrix(fea, mode, sigma=sigma)
    n = np.size(fea, 0)
    d1 = (w.sum(1)**(-0.5))[:,None]
    d2 = w.sum(0)**(-0.5)
    L = (d1 * w) * d2
    [u,s,v] = randomized_svd(L, n_components=k)
    if normalize:
        u = u * ((u**2).sum(1)**-0.5)[:,None]
        v = v * ((v**2).sum(1)**-0.5)[:,None]
    return u,s,v

def pick_representatives(fea, n_reps, mode='random'):
    if mode == 'random':
        n = fea.shape[0]
        idx = sample(range(n), n_reps)
        return fea[idx,:], idx
    elif mode == '++':
        idx = plusplus(fea, n_reps)
        return fea[idx,:], idx

def  plusplus(fea, n_reps):
    n1, n2 = fea.shape
    # first center
    centers_idx = np.zeros(n_reps, dtype=np.int32)
    idx = np.random.randint(0, n1)
    centers_idx[0] = idx
    if sp.isspmatrix(fea):
        new_center = fea[idx,:].toarray().reshape(1,n2)
    else: new_center = fea[idx,:].reshape(1,n2)

    closest_distances = eudist(fea, new_center, False).reshape(n1)
    for i in range(1, n_reps):
        idx = choice(n1, p=closest_distances/closest_distances.sum())
        centers_idx[i] = idx
        if sp.isspmatrix(fea):
            new_center = fea[idx,:].toarray().reshape(1,n2)
        else: new_center = fea[idx,:].reshape(1,n2)
        new_center_distances = eudist(fea, new_center, False).reshape(n1)
        closest_distances = np.minimum(closest_distances, new_center_distances)
        cumsum = cumdist_matrix(closest_distances)
    return centers_idx

def spectral_clustering(fea, k, mode, sigma=None):
    u,_,_ = _normalize_svd(fea, k, mode, sigma=sigma)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
    return kmeans.fit_predict(u)

def _landmark_bipartite_svd(fea, reps, k, affinity='gaussian', sigma=None):
    if affinity == 'gaussian':
        if sigma is None: raise Exception('affinity gaussian requires sigma be specified')
        w = eudist(fea, reps, False)
        w = 1 / np.exp


def svd_speed_test(mat):

    print('using numpy svd')
    t = time()
    numpysvd(mat, full_matrices=False, compute_uv=True)
    print('time elapsed for numpy svd: {}'.format(time() - t))

    print('using scipy sparse svds')
    t = time()
    svds(mat, 10)
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
    # gnd = content['gnd'].reshape(2000)
    a = time()
    _, idx1 = pick_representatives(fea, 1000, mode='random')
    b = time()
    print('time elapsed for random: {}'.format(b - a))
    a = time()
    _, idx2 = pick_representatives(fea, 1000, mode='++')
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
def main():
    test_plusplus('../data/news.mat')
    # mat = np.random.randn(2000, 2000)
    # svd_speed_test(mat)
    # content = scipy.io.loadmat('../data/circledata_50.mat', mat_dtype=True)
    # fea = content['fea']
    # gnd = content['gnd'].reshape(2000)
    # import matplotlib.pyplot as plt
    # plt.scatter(fea[:,0], fea[:,1], c=gnd.reshape(2000))
    # [u,s,v] = _normalize_svd(fea, 2, mode='gaussian', sigma=30)
    # print(u)
    # labels = spectral_clustering(fea, 2, mode='gaussian', sigma=30)
    # labels = [int(x) for x in labels]
    # gnd = [int(x) for x in gnd]
    # print(np.concatenate((labels, gnd), 1))
    # labels, diff = greedy_matching(labels, gnd)
    # print(labels)
    # print(2000 - diff)
    # plt.show()
if __name__ == '__main__':
    main()
