import numpy as np

from scipy import sparse


def dot(X, Y):
    if sparse.isspmatrix(X) and sparse.isspmatrix(Y):
        return X * Y
    elif sparse.isspmatrix(X) or sparse.isspmatrix(Y):
        return sparse.csr_matrix(X) * sparse.csr_matrix(Y)

    return np.asmatrix(X) * np.asmatrix(Y)
