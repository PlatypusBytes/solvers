import pytest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
except ImportError:
    cp = None
    cpsp = None

from solvers.linear_equations_solvers import (
    DenseDirectSolver,
    SparseDirectSolver,
    SparseDirectSolverLU,
    CGSolver,
    CGSolverGPU,
    GMRESSolver,
    BICSTABSolver,
)
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner, JacobiPreconditionerGPU
from tests.utils import has_cupy


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

@pytest.mark.parametrize("solver_cls", [SparseDirectSolver, SparseDirectSolverLU])
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


@pytest.mark.parametrize("solver_cls", [CGSolverGPU])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditionerGPU])
@pytest.mark.skipif(not has_cupy(), reason="CuPy not installed")
def test_iterative_solvers_GPU(spd_system, solver_cls, preconditioner):
    """
    Test that iterative solvers correctly solve a SPD system with and without a Jacobi preconditioner.
    Args:
        spd_system: Fixture providing a SPD system
        solver_cls: Solver class to test
        preconditioner: Preconditioner class or None
    """

    _, A_sparse, b, x_expected = spd_system

    A_sparse_gpu = cpsp.csc_matrix(A_sparse)
    b_gpu = cp.asarray(b)

    solver = solver_cls()

    if preconditioner is not None:
        pre_c = preconditioner().build(A_sparse_gpu)
    else:
        pre_c = None

    x = solver.solve(A_sparse_gpu, b_gpu, M=pre_c)
    np.testing.assert_allclose(x.get(), x_expected)
    np.testing.assert_allclose(A_sparse_gpu.dot(x).get(), b)


@pytest.fixture
def laplacian_system():
    """
    Creates a 1D SPD system, large enough for the iterative solvers to need many iterations.

    Returns:
        A_sparse: Sparse SPD matrix
        b: Right-hand side vector
        x_exact: Solution vector
    """
    n = 50
    A_sparse = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    b = np.ones(n)
    x_exact = spsolve(A_sparse.tocsc(), b)
    return A_sparse, b, x_exact


class TestInitialGuessAndIterations:
    """
    Initial guess (x0) and iteration count of the linear solvers.
    """

    @pytest.mark.parametrize("solver_cls", [CGSolver, GMRESSolver, BICSTABSolver])
    def test_iterations_recorded_per_solve(self, laplacian_system, solver_cls):
        """
        Every solve appends its number of iterations, in the order of the solves.
        """
        A_sparse, b, x_exact = laplacian_system
        solver = solver_cls()
        assert solver.iterations == []

        x = solver.solve(A_sparse, b)
        solver.solve(A_sparse, 2 * b)
        np.testing.assert_allclose(x, x_exact, rtol=1e-8)
        assert len(solver.iterations) == 2
        assert all(n > 0 for n in solver.iterations)

    @pytest.mark.parametrize("solver_cls", [CGSolver, GMRESSolver, BICSTABSolver])
    def test_solution_as_initial_guess_needs_no_iteration(self, laplacian_system, solver_cls):
        """
        Starting from the solution, the solver stops before its first iteration and returns it.
        """
        A_sparse, b, x_exact = laplacian_system
        solver = solver_cls()

        x = solver.solve(A_sparse, b, x0=x_exact.copy())
        np.testing.assert_allclose(x, x_exact, rtol=1e-12)
        assert solver.iterations == [0]

    def test_not_converged_raises_and_records_iterations(self, laplacian_system):
        """
        A solve that does not converge within maxiter raises, and its iterations are still recorded.
        """
        A_sparse, b, _ = laplacian_system
        solver = CGSolver(maxiter=3)

        with pytest.raises(RuntimeError, match="CG did not converge \(info=3\)"):
            solver.solve(A_sparse, b)
        assert solver.iterations == [3]

    @pytest.mark.parametrize("solver_cls", [SparseDirectSolver, SparseDirectSolverLU])
    def test_sparse_direct_solvers_ignore_initial_guess(self, laplacian_system, solver_cls):
        """
        The sparse direct solvers accept an initial guess and return the exact solution regardless.
        """
        A_sparse, b, x_exact = laplacian_system

        x = solver_cls().solve(A_sparse.tocsc(), b, x0=np.full_like(b, 1e3))
        np.testing.assert_allclose(x, x_exact, rtol=1e-12)

    def test_dense_direct_solver_ignores_initial_guess(self, laplacian_system):
        """
        The dense direct solver accepts an initial guess and returns the exact solution regardless.
        """
        A_sparse, b, x_exact = laplacian_system

        x = DenseDirectSolver().solve(A_sparse.toarray(), b, x0=np.full_like(b, 1e3))
        np.testing.assert_allclose(x, x_exact, rtol=1e-12)

    @pytest.mark.skipif(not has_cupy(), reason="CuPy not installed")
    def test_gpu_initial_guess_and_iterations(self, laplacian_system):
        """
        CGSolverGPU records its iterations and stops after its first iteration from the solution:
        unlike scipy, CuPy checks convergence only at the end of an iteration.
        """
        A_sparse, b, x_exact = laplacian_system
        A_gpu = cpsp.csc_matrix(A_sparse)
        b_gpu = cp.asarray(b)
        solver = CGSolverGPU()

        x = solver.solve(A_gpu, b_gpu)
        x_restart = solver.solve(A_gpu, b_gpu, x0=cp.asarray(x_exact))
        np.testing.assert_allclose(x.get(), x_exact, rtol=1e-8)
        np.testing.assert_allclose(x_restart.get(), x_exact, rtol=1e-12)
        assert solver.iterations[0] > 1
        assert solver.iterations[1] == 1
