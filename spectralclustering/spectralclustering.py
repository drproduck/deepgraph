import numpy as np
from numpy.random import choice
from numpy.linalg import svd as numpysvd
import scipy.io

from scipy.sparse.linalg import svds
from scipy.linalg.decomp_svd import svd as scipysvd
from sklearn.utils.extmath import randomized_svd

from sklearn.cluster import KMeans, k_means_
from time import time
from utils.io import make_weight_matrix, greedy_matching, accuracy, best_map
from utils.matop import eudist, cumdist_matrix
from random import sample, choices
import scipy.sparse as sp
import scipy.io as scio

def _symmetric_laplacian(fea, k, mode, sigma=None):
    if mode == 'gaussian' and sigma is None:
        raise Exception('mode Gaussian requires sigma specified')
        w = make_weight_matrix(fea, mode, sigma=sigma)
    elif mode == 'cosine':
        ""
    d1 = (w.sum(1)**(-0.5))[:,None]
    d2 = w.sum(0)**(-0.5)
    L = (d1 * w) * d2
    return L

def landmark_bipartite_laplacian(fea, reps, k, affinity='gaussian', sparsity=3, sigma=None):
    if affinity == 'gaussian':
        if sigma is None: raise Exception('affinity gaussian requires sigma be specified')
        w = eudist(fea, reps, False)
        w = 1.0 / np.exp(w / (2 * sigma**2))
        if sparsity is not None:
            w, closest_rep = nearest_k_sparsity(w, sparsity, 'max')
        n1, n2 = w.shape
        d1 = w.sum(1).A1
        d1 = np.maximum(d1, 1e-10)
        d2 = w.sum(0).A1
        d2 = np.maximum(d2, 1e-10)
        d1 = sp.spdiags(d1**(-0.5),diags=0,m=n1,n=n1)
        d2 = sp.spdiags(d2**(-0.5),diags=0,m=n2,n=n2)
        L = d1 * w * d2
        return L, closest_rep

def _svd_embedding(L, k, normalize=True, remove_first=True):
    if sp.isspmatrix(L):
        u,s,v = svds(L, k=k)
    else: u,s,v = randomized_svd(L, n_components=k)
    v = v.T
    if remove_first:
        u = u[:,1:]
        v = v[:,1:]
        s = s[1:]
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
    elif mode == 'kmeans':
        kmeans = KMeans(n_reps, init='random', n_init=10, max_iter=1, n_jobs=-1)
        kmeans.fit(fea)
        return kmeans.cluster_centers_

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
    closest_distances[idx] = 0
    for i in range(1, n_reps):
        idx = choices(range(n1), weights=closest_distances, k=1)[0]
        centers_idx[i] = idx
        if sp.isspmatrix(fea):
            new_center = fea[idx,:].toarray().reshape(1,n2)
        else: new_center = fea[idx,:].reshape(1,n2)
        distances_to_new_center = eudist(fea, new_center, False).reshape(n1)
        distances_to_new_center[idx] = 0
        closest_distances = np.minimum(closest_distances, distances_to_new_center)
        # print(closest_distances[closest_distances < 0])
    return centers_idx

def nearest_k_sparsity(w, sparsity = 3, max_or_min='max', save=False, toarray=False):
    """convert a n*m matrix to a sparse n*m matrix where only {sparsity} values in each row are kept
        :returns the sparse (CSC) matrix and cols: the original indices of maximum values"""
    n1, n2 = w.shape
    cols = np.zeros((sparsity, n1), dtype=np.int64)
    vals = np.zeros((sparsity, n1))
    row_enum = np.arange(n1)
    rows = np.tile(row_enum, (sparsity, 1))
    if max_or_min == 'max':
        for i in range(sparsity):
            col_argmax = np.argmax(w, 1).flat
            vals[i] = w[row_enum, col_argmax]
            w[row_enum, col_argmax] = np.NINF
            cols[i] = col_argmax
    elif max_or_min == 'min':
        for i in range(sparsity):
            col_argmin = np.argmin(w, 1)
            w[row_enum, col_argmin] = np.PINF
            cols[i] = col_argmin
    sw = sp.coo_matrix((vals.flat, (rows.flat, cols.flat)), shape=(n1, n2), dtype=np.float64).tocsr()
    if save: scio.savemat('circledata_sparse',{'fea': sw})
    if toarray: return sw.toarray()
    print(cols)
    print(rows)
    return sw, cols


def spectral_clustering(fea, k, affinity, sigma=None):
    L = _symmetric_laplacian(fea, k, affinity, sigma=sigma)
    u,_,_ = _svd_embedding(L, k)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, n_jobs=-1)
    return kmeans.fit_predict(u)

def bipartite_clustering(fea, k, affinity, n_reps=500, select_method='++',
                         sparsity=3, use_embedding='v', sigma=None):
    n1, n2 = fea.shape
    reps,_ = pick_representatives(fea, n_reps, select_method)
    L, nearest_rep = landmark_bipartite_laplacian(fea, reps, k, affinity, sparsity=sparsity, sigma=sigma)
    u,s,v = _svd_embedding(L, k)
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100, n_jobs=-1)
    if use_embedding=='u':
        return kmeans.fit_predict(u)
    elif use_embedding=='v':
        rep_label = kmeans.fit_predict(v)
        labels = np.zeros(n1, dtype=np.int32)
        for i in range(n1):
            labels[i] = rep_label[nearest_rep[0,i]]
        return labels
    elif use_embedding=='uv':
        return kmeans.fit_predict(np.concatenate((u,v), axis=0))

