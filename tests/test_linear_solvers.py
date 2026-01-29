import pytest
import numpy as np
from scipy import sparse

from solvers.linear_equations_solvers import (
    DenseDirectSolver,
    SparseDirectSolver,
    SparseDirectSolverInv,
    CGSolver,
    GMRESSolver,
    BICSTABSolver,
)
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner


@pytest.fixture
def spd_system():
    """
    Creates a small SPD system usable by dense and sparse solvers.

    Returns:
        A: Dense SPD matrix
        A_sparse: Sparse SPD matrix
        b: Right-hand side vector
        x_expected: Expected solution vector
    """
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    x_exact = np.linalg.solve(A, b)
    A_sparse = sparse.csr_matrix(A)
    return A, A_sparse, b, x_exact

@pytest.mark.parametrize("solver_cls", [DenseDirectSolver])
def test_dense_direct_solver_solves_and_caches(spd_system, solver_cls):
    """
    Test that DenseDirectSolver correctly solves a SPD system and caches the matrix.
    Args:
        spd_system: Fixture providing a SPD system
        solver_cls: Solver class to test
    """
    A_dense, _, b, x_expected = spd_system
    solver = solver_cls()

    x_first = solver.solve(A_dense, b.copy())
    np.testing.assert_allclose(x_first, x_expected)

    preconditioner = JacobiPreconditioner().build(sparse.csc_matrix(A_dense))
    with pytest.warns(UserWarning, match="Preconditioner is ignored"):
        solver.solve(A_dense, b, M=preconditioner)

@pytest.mark.parametrize("solver_cls", [SparseDirectSolver, SparseDirectSolverInv])
def test_sparse_direct_solver_solves_and_caches(spd_system, solver_cls):
    """
    Test that sparse linear solvers correctly solve a SPD system and cache the matrix.
    Args:
        spd_system: Fixture providing a SPD system
        solver_cls: Solver class to test
    """
    _, A_sparse, b, x_expected = spd_system
    solver = solver_cls()

    x_first = solver.solve(A_sparse, b.copy())
    np.testing.assert_allclose(x_first, x_expected)

    preconditioner = JacobiPreconditioner().build(sparse.csc_matrix(A_sparse))
    with pytest.warns(UserWarning, match="Preconditioner is ignored"):
        solver.solve(A_sparse, b, M=preconditioner)

@pytest.mark.parametrize("solver_cls", [CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
def test_iterative_solvers_handle_spd_system(spd_system, solver_cls, preconditioner):
    """
    Test that iterative solvers correctly solve a SPD system with and without a Jacobi preconditioner.
    Args:
        spd_system: Fixture providing a SPD system
        solver_cls: Solver class to test
        preconditioner: Preconditioner class or None
    """

    _, A_sparse, b, x_expected = spd_system
    solver = solver_cls()

    if preconditioner is not None:
        pre_c = preconditioner().build(A_sparse)
    else:
        pre_c = None
    x = solver.solve(A_sparse, b, M=pre_c)
    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(A_sparse.dot(x), b)
