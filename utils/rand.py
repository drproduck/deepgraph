from random import random
import numpy as np
from math import floor


class alias_sampling():
    def __init__(self, array):
        s = np.shape(array)
        self.shape = s
        if len(s) != 1 and len(s) != 2: raise Exception('Invalid shape, only array of dimension 1 or 2 is allowed')
        self.n_face = s[0] if len(s) == 1 else s[1]
        self.weights_table = array
        self.prob_table = np.zeros(shape=s, dtype=np.float64)
        self.alias_table = np.zeros(shape=s, dtype=np.int64)
        if len(s) == 1: self.dim1_sampling(self.weights_table, self.prob_table,
                                           self.alias_table)
        elif len(s) == 2: self.dim2_sampling()


    def dim1_sampling(self, weights, prob, alias):
        alpha = (1.0 * self.n_face) / weights.sum(0)
        weights *= alpha
        small = list()
        large = list()
        for i in range(weights.shape[0]):
            small.append(i) if weights[i] < 1 else large.append(i)
        while len(small) != 0 and len(large) != 0:
            s = small.pop()
            l = large.pop()
            prob[s] = weights[s]
            alias[s] = l
            weights[l] = weights[s] + weights[l] - 1
            small.append(l) if weights[l] < 1 else large.append(l)

        while len(large) != 0:
            g = large.pop()
            prob[g] = 1
        while len(small) != 0:
            g = small.pop()
            prob[g] = 1

    def dim2_sampling(self):
        for i in range(self.weights_table.shape[0]):
            self.dim1_sampling(self.weights_table[i], self.prob_table[i],
                               self.alias_table[i])

    def sample(self, index=None):
        if len(self.shape) == 2 and index is None: raise Exception('index required for table of multiple distributions')
        if len(self.shape) == 1:
            prob = self.prob_table
            alias = self.alias_table
        elif len(self.shape) == 2:
            prob = self.prob_table[index,:]
            alias = self.alias_table[index,:]
        face = floor(random()*self.n_face)
        return face if random() < prob[face] else alias[face]

def main():
    gen = alias_sampling(np.array([[1,2,3,4],[4,3,2,1]], dtype=np.float64))
    x = np.array([gen.sample(0) for i in range(10000)])
    y = np.array([gen.sample(1) for i in range(10000)])
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(121)
    plt.hist(x)
    plt.subplot(122)
    plt.hist(y)
    plt.show()

if __name__ == '__main__':
    main()