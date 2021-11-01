from pypardiso import factorized
import numpy as np
import scipy.sparse


def sp_solve_sparse_result(A,b):
    """
    Sparse solver with a sparse result
    :param A:
    :param b:
    :return:
    """
    # factorize matrix A
    A_factorize = factorized(A)

    data_segs = []
    row_segs = []
    col_segs = []
    # loop over columns of matrix b
    for j in range(b.shape[1]):

        # solve Ax[j] = b[j]
        bj = np.asarray(b[:, j].todense()).ravel()
        xj = A_factorize(bj)

        # add x[j] to sparse csc structure
        w = np.flatnonzero(xj)
        segment_length = w.shape[0]
        row_segs.append(w)
        col_segs.append(np.full(segment_length, j, dtype=int))
        data_segs.append(np.asarray(xj[w], dtype=A.dtype))

    # combine data, rows and columns to sparse matrix
    sparse_data = np.concatenate(data_segs)
    sparse_row = np.concatenate(row_segs)
    sparse_col = np.concatenate(col_segs)
    x = A.__class__((sparse_data, (sparse_row, sparse_col)),
                    shape=b.shape, dtype=A.dtype)

    return x

def inv(A):
    """
    Calculate inverse of matrix A
    :param A:
    :return:
    """

    # construct sparse identity matrix with shape and type of A
    I = scipy.sparse.construct.eye(A.shape[0], A.shape[1],
                               dtype=A.dtype, format=A.format)

    # calculate inverse A
    inverse_A = sp_solve_sparse_result(A, I)

    return inverse_A


