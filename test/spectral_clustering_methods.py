import numpy.testing as npt
import numpy as np
import unittest
from spectralclustering.spectralclustering import  nearest_k_sparsity
import scipy.sparse as sp

class test_spectral_clustering():
    def test_repeat(self):
        row = np.array([0,1,2,3,4], dtype=np.float64)
        row3 = np.tile(row, (3,1))
        npt.assert_almost_equal(row3, np.array([[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]))

    def test_nearest_k_sparsity(self):
        w = np.arange(15).reshape(3,5)
        w = w * 1.0
        sw = nearest_k_sparsity(w, 3)
        sw = sw.toarray()
        target = np.array([[0,0,2,3,4],[0,0,7,8,9],[0,0,12,13,14]], dtype=np.float64)
        npt.assert_almost_equal(sw, target)


def main():
    w = np.arange(15).reshape(3,5)
    w = w * 1.0
    sw = nearest_k_sparsity(w, 3)
    sw.toarray()
    print(sw)
if __name__ == '__main__':
    # main()
    npt.run_module_suite()