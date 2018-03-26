import random
import numpy as np
import matop


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


def graph_feeder(adjmatrix_path, window_size, cumulative=True):
    adjmatrix = np.loadtxt(adjmatrix_path)
    n = np.size(adjmatrix, 0)
    m = np.size(adjmatrix, 1)
    if not n == m: raise Exception('has to be square matrix')
    skip_window = 2 * window_size
    if not cumulative:
        import itertools
        adjmatrix = np.array(list(itertools.accumulate(adjmatrix.transpose()))).transpose()

    while True:
        od = list(range(n))
        for node in od:
            prev_node = node
            for _ in range(skip_window):
                next_node = random.choices(range(n), weights=adjmatrix[prev_node], k=1)[0]
                yield (node, next_node)
                prev_node = next_node

def batch_feeder(path, mode, batch_size, window_size):
    if mode == 'text':
        feed = text_feeder(path, window_size)
    # assuming that graph is True
    elif mode == 'graph':
        feed = graph_feeder(path, window_size, cumulative=False)
    elif mode == 'cum_graph':
        feed = graph_feeder(path, window_size, cumulative=True)
    else: raise Exception('unsupported mode')

    context = np.ndarray([batch_size], dtype=np.int32)
    targets = np.ndarray([batch_size, 1], dtype=np.int32)
    while True:
        for i in range(batch_size):
            context[i], targets[i, 0] = next(feed)
        yield (context, targets)


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
