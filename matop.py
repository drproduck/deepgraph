import numpy as np
import time
def sloweudist(A, B):
    n = np.size(A, 0)
    m = np.size(B,0)
    x = np.zeros(shape=[n,m])
    for i in range(n):
        for j in range(m):
            x[i,j] = np.linalg.norm(A[i,:] - B[j,:])
    return x

def eudist(A, B, sqrted=True):
    n = np.size(A, 0)
    m = np.size(B, 0)
    a = np.sum(A ** 2, 1).reshape([n,1])
    AA = np.repeat(a, m, 1)
    b = np.sum(B ** 2, 1).reshape([1,m])
    BB = np.repeat(b, n, 0)
    AB = 2 * np.matmul(A, np.transpose(B))
    if sqrted:
        return (AA - AB + BB) ** 0.5
    else: return (AA - AB + BB)


def main():
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print(eudist(a,b,False))

    a = np.random.normal(0,1, [2000, 2000])
    b = np.random.normal(0,1, [4000, 2000])
    start = time.time()
    x = eudist(a,b, True)
    stop = time.time()
    print(x)
    print(stop - start)
    start = time.time()
    x = sloweudist(a,b)
    stop = time.time()
    print(x)
    print(stop - start)

if __name__ == "__main__":
    main()
