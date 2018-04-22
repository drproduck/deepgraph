import numpy.testing as npt
import numpy as np
import unittest
import spectralclustering.spectralclustering as sc
import utils.io
import scipy
import utils

class test_spectral_clustering():
    def test_symmetric_laplacian(self):
        w = np.arange(15).reshape(3,5)
        actual = np.array( [[0.        , 0.0745356 , 0.13801311, 0.19364917, 0.24343225],
                            [0.21821789, 0.23904572, 0.25819889, 0.27602622, 0.29277002],
                            [0.33333333, 0.33471934, 0.3380617 , 0.34258008, 0.3478328 ]])

        npt.assert_almost_equal(sc.symmetric_laplacian(w), actual)

    def test_repeat(self):
        row = np.array([0,1,2,3,4], dtype=np.float64)
        row3 = np.tile(row, (3,1))
        npt.assert_almost_equal(row3, np.array([[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]))

    def test_nearest_k_sparsity(self):
        w = np.array([[4,2,1,3,0],[6,9,8,7,5],[10,13,14,11,12]])
        w = w * 1.0
        sw,cols = sc.nearest_k_sparsity(w, 3)
        sw = sw.toarray()
        target = np.array([[4,2,0,3,0],[0,9,8,7,0],[0,13,14,0,12]], dtype=np.float64)
        npt.assert_almost_equal(sw, target)
        npt.assert_almost_equal(cols, np.array([[0,1,2],[3,2,1],[1,3,4]]))

    def test_eudist(self):
        A = np.array([[1,2,3,4],[5,6,7,8]])
        B = np.array([[9,10,11,12]])
        resfalse = utils.matop.eudist(A,B,False)
        restrue = utils.matop.eudist(A,B,True)
        targetfalse = np.array([[256],[64]])
        targettrue = np.array([[16],[8]])
        npt.assert_almost_equal(restrue, targettrue)
        npt.assert_almost_equal(resfalse, targetfalse)

    def test_accuracy(self):
        a = np.array([1,2,3,4,5])
        b =  np.array([[1],[2],[3],[4],[6]])
        npt.assert_equal(utils.io.accuracy(a,b), 0.8)

    def test_greedy_matching(self):
        a = np.array([1,1,0,0,0,1])
        b = np.array([2,2,1,1,1,2])
        a,_ = utils.io.greedy_matching(a,b)
        npt.assert_equal(a, b)

def main():
    w = np.arange(15).reshape(3,5)
    w = w * 1.0
    sw = sc.nearest_k_sparsity(w, 3)
    sw.toarray()
    print(sw)
if __name__ == '__main__':
    # main()
    npt.run_module_suite()