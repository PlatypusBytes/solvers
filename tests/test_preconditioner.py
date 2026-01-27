import pytest

import numpy as np
from scipy.sparse import csr_matrix

from solvers.utils import PreConditioner


@pytest.fixture
def test_matrix():
    """
    SPD matrix with known SSOR/Jacobi behavior
    """
    A = np.array([
        [ 4.0, -1.0,  0.0],
        [-1.0,  4.0, -1.0],
        [ 0.0, -1.0,  4.0]
    ])
    return csr_matrix(A)

@pytest.fixture
def test_vector():
    return np.array([1., 1., 1.])


def test_none_preconditioner(test_matrix):
    """
    Test that the NONE preconditioner returns None.
    """
    A = test_matrix
    M = PreConditioner.NONE.apply(A)
    assert M is None


def test_jacobi(test_matrix, test_vector):
    """
    Test that the JACOBI preconditioner correctly applies the inverse of the diagonal.
    """

    A = test_matrix
    v = test_vector

    M = PreConditioner.JACOBI.apply(A)
    result = M @ v

    diag = np.diag(A.toarray())
    expected = v / diag

    np.testing.assert_allclose(result, expected)

def test_ssor(test_matrix, test_vector):
    """
    SSOR preconditioner test.

    This test checks that the SSOR preconditioner matches the expected behavior.
    """
    A = test_matrix
    v = test_vector

    M = PreConditioner.SSOR.apply(A, omega=1.0)
    result = M @ v

    # decomposition of M (for verification -> omega = 1)
    D = np.diag(np.diag(A.toarray()))
    L = np.tril(A.toarray(), k=-1)
    U = np.triu(A.toarray(), k=1)
    DL = D + L
    DU = D + U

    M_exact = np.linalg.inv(DU) @ np.linalg.inv(D) @ np.linalg.inv(DL)

    np.testing.assert_allclose(result, M_exact @ v, atol=1e-6)

def test_ilu(test_matrix):
    """
    Incomplete LU preconditioner test.

    This test checks that the ILU preconditioner approximates the inverse of the matrix.
    """
    A = test_matrix

    M =  PreConditioner.ILU.apply(A, drop_tol=0.0, fill_factor=10)
    I_approx = np.column_stack([M @ np.eye(3)[:, i] for i in range(3)])
    A_inv = np.linalg.inv(A.toarray())
    np.testing.assert_allclose(I_approx, A_inv, atol=1e-6)
