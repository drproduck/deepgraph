import numpy as np
from numpy.linalg import svd as numpysvd
import scipy.io
from scipy.sparse.linalg import svds
from scipy.linalg.decomp_svd import svd as scipysvd
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
from time import time
from utils import make_weight_matrix

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

def spectral_clustering(fea, k, mode, sigma=None):
    u,_,_ = _normalize_svd(fea, k, mode, sigma=sigma)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
    return kmeans.fit_predict(u)

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

def main():
    # mat = np.random.randn(2000, 2000)
    # svd_speed_test(mat)
    content = scipy.io.loadmat('/home/drproduck/Documents/circledata_50.mat')
    fea = content['fea']
    gnd = content['gnd']
    # [u,s,v] = _normalize_svd(fea, 2, mode='gaussian', sigma=30)
    # print(u)
    labels = spectral_clustering(fea, 2, mode='gaussian', sigma=30)
    # print(np.concatenate((labels, gnd), 1))
    print(labels)
    print(gnd)
if __name__ == '__main__':
    main()
