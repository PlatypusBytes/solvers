import pytest

import numpy as np
from scipy.sparse import csr_matrix

from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner, JacobiPreconditionerGPU
from tests.utils import has_cupy


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


def test_jacobi(test_matrix, test_vector):
    """
    Test that the JACOBI preconditioner correctly applies the inverse of the diagonal.

    Args:
        test_matrix: Fixture providing a test matrix
        test_vector: Fixture providing a test vector
    """
    A = test_matrix
    v = test_vector

    M = JacobiPreconditioner().build(A)
    result = M @ v

    diag = np.diag(A.toarray())
    expected = v / diag

    np.testing.assert_allclose(result, expected)


@pytest.mark.skipif(not has_cupy(), reason="CuPy not installed")
def test_jacobiGPU(test_matrix, test_vector):
    """
    Test that the JACOBI preconditioner correctly applies the inverse of the diagonal.

    Args:
        test_matrix: Fixture providing a test matrix
        test_vector: Fixture providing a test vector
    """
    A = test_matrix
    v = test_vector

    M = JacobiPreconditionerGPU().build(A)
    result = M @ v

    diag = np.diag(A.toarray())
    expected = v / diag

    np.testing.assert_allclose(result, expected)


def test_ssor(test_matrix, test_vector):
    """
    SSOR preconditioner test.

    This test checks that the SSOR preconditioner matches the expected behavior.

    Args:
        test_matrix: Fixture providing a test matrix
        test_vector: Fixture providing a test vector
    """
    A = test_matrix
    v = test_vector

    M = SSORPreconditioner(omega=1.0).build(A)
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

    Args:
        test_matrix: Fixture providing a test matrix
        test_vector: Fixture providing a test vector
    """
    A = test_matrix

    M =  ILUPreconditioner(drop_tol=0.0, fill_factor=10).build(A)
    A_inv_approx = np.column_stack([M @ np.eye(3)[:, i] for i in range(3)])
    A_inv = np.linalg.inv(A.toarray())
    np.testing.assert_allclose(A_inv_approx, A_inv, atol=1e-6)
