import pytest

import numpy as np
from scipy import sparse

from solvers.bathe_solver import BatheSolver
from solvers.utils import LumpingMethod
from tests.utils import set_matrices_as_sparse, set_matrices_as_np_array
from solvers.base_solver import Force, State
from solvers.linear_equations_solvers import (SparseDirectSolver, SparseDirectSolverInv, DenseDirectSolver,
    CGSolver, GMRESSolver, BICSTABSolver)
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner


@pytest.fixture
def bathe_basic_setup():
    """
    Basic setup for Bathe solver tests.
    """
    # example from bathe
    M = [[2, 0], [0, 1]]
    K = [[6, -2], [-2, 4]]
    C = [[0, 0], [0, 0]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M_mat = sparse.csc_matrix(np.array(M))
    K_mat = sparse.csc_matrix(np.array(K))
    C_mat = sparse.csc_matrix(np.array(C))
    F_mat = sparse.csc_matrix(np.array(F))

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2

    return M_mat, K_mat, C_mat, F_mat, n_steps, time, number_eq


@pytest.fixture
def full_matrix_setup():
    """
    Basic setup for Bathe solver tests with full matrices.
    """
    M = [[1, 1], [0.25, 0.75]]
    K = [[6, -2], [-2, 4]]
    C = [[0, 0], [0, 0]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M_mat = sparse.csc_matrix(np.array(M))
    K_mat = sparse.csc_matrix(np.array(K))
    C_mat = sparse.csc_matrix(np.array(C))
    F_mat = sparse.csc_matrix(np.array(F))

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2
    return M_mat, K_mat, C_mat, F_mat, n_steps, time, number_eq


@pytest.fixture
def full_matrix_damping_setup():
    """
    Basic setup for Bathe solver tests with full matrices and damping.
    """
    M = [[1, 1], [0.25, 0.75]]
    K = [[6, -2], [-2, 4]]
    C = [[0.25, 0.15], [0.15, 0.25]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M_mat = sparse.csc_matrix(np.array(M))
    K_mat = sparse.csc_matrix(np.array(K))
    C_mat = sparse.csc_matrix(np.array(C))
    F_mat = sparse.csc_matrix(np.array(F))

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2

    return M_mat, K_mat, C_mat, F_mat, n_steps, time, number_eq


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverInv, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("lumping_method", [LumpingMethod.NONE, LumpingMethod.RowSum])
def test_bathe_sparse(bathe_basic_setup, linear_solver, preconditioner, lumping_method):
    """
    Test Bathe solver with different linear solvers and preconditioners using sparse matrices and damping.

    Args:
        bathe_basic_setup: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
        lumping_method: Lumping method
    """
    M, K, C, F, n_steps, time, number_eq = bathe_basic_setup
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = BatheSolver(Force(), State(), linear_solver=linear_solver(),
                      preconditioner=prec, lumping_method=lumping_method)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    expected_results = np.array(
        [
            [0.00000000e+00, 0.00000000e+00],
            [2.06118743e-03, 3.83755250e-01],
            [3.70084511e-02, 1.41679373e+00],
            [1.74720597e-01, 2.78827698e+00],
            [4.86820747e-01, 4.09955407e+00],
            [1.00012810e+00, 4.99686663e+00],
            [1.66385091e+00, 5.28350017e+00],
            [2.34651973e+00, 4.97216771e+00],
            [2.86729415e+00, 4.26118800e+00],
            [3.05261241e+00, 3.44650610e+00],
            [2.79848629e+00, 2.80502692e+00],
            [2.11529439e+00, 2.49432910e+00],
            [1.13720739e+00, 2.50615265e+00],
        ]
    )
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(expected_results, 2)
    )


@pytest.mark.parametrize("linear_solver", [DenseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("lumping_method", [LumpingMethod.NONE, LumpingMethod.RowSum])
def test_bathe_dense(bathe_basic_setup, linear_solver, lumping_method):
    """
    Test Bathe solver with different linear solvers using dense matrices or sparse matrices.

    Args:
        bathe_basic_setup: Fixture setting up matrices
        linear_solver: Linear solver class
        lumping_method: Lumping method
    """
    M, K, C, F, n_steps, time, number_eq = bathe_basic_setup

    M, K, C, F = set_matrices_as_np_array(M, K, C, F)

    res = BatheSolver(Force(), State(), linear_solver=linear_solver(),
                      preconditioner=None, lumping_method=lumping_method)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    expected_results = np.array(
        [
            [0.00000000e+00, 0.00000000e+00],
            [2.06118743e-03, 3.83755250e-01],
            [3.70084511e-02, 1.41679373e+00],
            [1.74720597e-01, 2.78827698e+00],
            [4.86820747e-01, 4.09955407e+00],
            [1.00012810e+00, 4.99686663e+00],
            [1.66385091e+00, 5.28350017e+00],
            [2.34651973e+00, 4.97216771e+00],
            [2.86729415e+00, 4.26118800e+00],
            [3.05261241e+00, 3.44650610e+00],
            [2.79848629e+00, 2.80502692e+00],
            [2.11529439e+00, 2.49432910e+00],
            [1.13720739e+00, 2.50615265e+00],
        ]
    )
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(expected_results, 2)
    )


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverInv, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
def test_bathe_sparse_full(full_matrix_setup, linear_solver, preconditioner):
    """
    Test Bathe solver with different linear solvers and preconditioners using sparse full matrices.

    Args:
        full_matrix_setup: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
        lumping_method: Lumping method
    """
    M, K, C, F, n_steps, time, number_eq = full_matrix_setup
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = BatheSolver(Force(), State(), linear_solver=linear_solver(),
                      preconditioner=prec, lumping_method=LumpingMethod.RowSum)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    expected_results = np.array(
        [
            [0.00000000e+00, 0.00000000e+00],
            [2.06118743e-03, 3.83755250e-01],
            [3.70084511e-02, 1.41679373e+00],
            [1.74720597e-01, 2.78827698e+00],
            [4.86820747e-01, 4.09955407e+00],
            [1.00012810e+00, 4.99686663e+00],
            [1.66385091e+00, 5.28350017e+00],
            [2.34651973e+00, 4.97216771e+00],
            [2.86729415e+00, 4.26118800e+00],
            [3.05261241e+00, 3.44650610e+00],
            [2.79848629e+00, 2.80502692e+00],
            [2.11529439e+00, 2.49432910e+00],
            [1.13720739e+00, 2.50615265e+00],
        ]
    )
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(expected_results, 2)
    )


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverInv, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
def test_bathe_damping(full_matrix_damping_setup, linear_solver, preconditioner):
    """
    Test Bathe solver with different linear solvers and preconditioners using sparse full matrices and damping.

    Args:
        full_matrix_damping_setup: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
    """
    M, K, C, F, n_steps, time, number_eq = full_matrix_damping_setup
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = BatheSolver(Force(), State(), linear_solver=linear_solver(),
                      preconditioner=prec, lumping_method=LumpingMethod.RowSum)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    expected_results = np.array(
        [
            [0.00000000e+00, 0.00000000e+00],
            [1.63586304e-05, 3.76939154e-01],
            [1.76324610e-02, 1.35533885e+00],
            [1.11524199e-01, 2.60529266e+00],
            [3.48662646e-01, 3.74848682e+00],
            [7.60516928e-01, 4.48344428e+00],
            [1.31225415e+00, 4.67566321e+00],
            [1.89918182e+00, 4.38323241e+00],
            [2.37359393e+00, 3.81322407e+00],
            [2.59288951e+00, 3.22945309e+00],
            [2.47112334e+00, 2.84872601e+00],
            [2.01433846e+00, 2.76474086e+00],
            [1.32574590e+00, 2.92620792e+00],
        ]
    )
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(expected_results, 2)
    )
