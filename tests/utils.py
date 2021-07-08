from scipy import sparse


def set_matrices_as_sparse(M, K, C, F):
    M = sparse.csc_matrix(M)
    K = sparse.csc_matrix(K)
    C = sparse.csc_matrix(C)
    F = sparse.csc_matrix(F)

    return M, K, C, F


def set_matrices_as_np_array(M, K, C, F):
    if sparse.issparse(M):
        M = M.toarray()
    if sparse.issparse(K):
        K = K.toarray()
    if sparse.issparse(C):
        C = C.toarray()
    if sparse.issparse(M):
        F = F.toarray()

    return M, K, C, F