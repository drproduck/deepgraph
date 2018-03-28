import random
import numpy as np
import matop
import scipy.sparse as sparse

def text_feeder(path, window_size):
    f = open(path, 'rb')
    word_list = [word for line in f for word in line.split()]
    word_set = set(word_list)
    word_to_idx = dict()
    idx = 0
    for word in word_set:
        word_to_idx[word] = idx
        idx += 1
    idx_list = [word_to_idx[word] for word in word_list]
    while True:
        for i in range(window_size, len(word_list) - window_size):
            for j in range(window_size * 2 + 1):
                yield ((idx_list[i], idx_list[i - window_size + j]))


def graph_feeder(adjmatrix_path=None, adjmatrix=None, window_size=10, cumulative=True):
    if not adjmatrix_path is None and adjmatrix is None:
        adjmatrix = np.loadtxt(adjmatrix_path)
    elif (adjmatrix_path is None and adjmatrix is None) or (adjmatrix_path is not None and adjmatrix is not None):
        raise Exception('either adjmatrix_path or adjmatrix but not both has to be specified')
    n = np.size(adjmatrix, 0)
    m = np.size(adjmatrix, 1)
    if not n == m: raise Exception('has to be square matrix')
    skip_window = 2 * window_size
    if not cumulative:
        import itertools

    if not sparse.isspmatrix(adjmatrix):
        adjmatrix = np.array(list(itertools.accumulate(adjmatrix.transpose()))).transpose()
        while True:
            od = np.arange(n)
            random.shuffle(od)
            for node in od:
                prev_node = node
                for _ in range(skip_window):
                    next_node = random.choices(range(n), cum_weights=adjmatrix[prev_node], k=1)[0]
                    yield (node, next_node)
                    prev_node = next_node
    else:
        print('This is sparse')
        while True:
            od = np.arange(n)
            random.shuffle(od)
            for node in od:
                prev_node = node
                for _ in range(skip_window):
                    _, c, v = sparse.find(adjmatrix[prev_node])
                    next_node = random.choices(c, weights=v, k=1)[0]
                    yield(node, next_node)
                    prev_node = next_node



def batch_feeder(path=None, mat=None, mode='text', batch_size=128, window_size=10):
    if mode == 'text':
        feed = text_feeder(path, window_size)
    elif mode == 'graph':
        feed = graph_feeder(path, mat, window_size, cumulative=False)
    elif mode == 'cum_graph':
        feed = graph_feeder(path, mat, window_size, cumulative=True)
    else: raise Exception('unsupported mode')

    context = np.ndarray([batch_size], dtype=np.int32)
    targets = np.ndarray([batch_size, 1], dtype=np.int32)
    while True:
        for i in range(batch_size):
            context[i], targets[i, 0] = next(feed)
        yield (context, targets)

def greedy_matching(labels, targets, in_place=False):
    """match labels to target labels, so that difference is minimized"""
    from collections import Counter
    labels_count = Counter(labels)
    targets_count = Counter(targets)
    labels_sort = labels_count.most_common()
    targets_sort = targets_count.most_common()
    mapping = dict()
    diff = 0
    print(labels_sort)
    print(targets_sort)
    for x, y in zip(labels_sort, targets_sort):
        print(x[0],x[1],y[0],y[1])
        mapping[x[0]] = y[0]
        diff += abs(x[1] - y[1])
        print(mapping)
    return [mapping[x] for x in labels], diff


def make_weight_matrix(fea, mode, **kwargs):
    assert type(fea) == np.ndarray
    if mode == 'gaussian':
        w = matop.eudist(fea, fea, False)
        w = np.exp(-w / (2 * kwargs['sigma'] ** 2))
    elif mode == 'cosine':
        fea = fea / (np.sum(fea ** 2, 1) ** 0.5).reshape([np.size(fea, 0), 1])
        w = fea * fea.transpose()
    else:
        raise Exception('unsupported mode: {}'.format(mode))
    return w
