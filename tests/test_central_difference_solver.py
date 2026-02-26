import pytest
import numpy as np
from scipy import sparse

from solvers.base_solver import Force, State
from solvers.central_difference_solver import CentralDifferenceSolver
from solvers.utils import LumpingMethod
from tests.utils import set_matrices_as_sparse, set_matrices_as_np_array, ALL_LINEAR_SOLVERS, ALL_PRECONDITIONERS


@pytest.fixture
def setup_module():
    """
    Setup matrices for testing central difference solver.
    Example from Bathe Chp: 9.2.1 (pg 770).

    Returns:
        M, K, C, F: Mass, Stiffness, Damping matrices and Force vector
        n_steps: number of time steps
        time: time vector
        number_eq: number of equations
    """

    # example from bathe
    M = [[2, 0], [0, 1]]
    K = [[6, -2], [-2, 4]]
    C = [[0, 0], [0, 0]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M = sparse.csc_matrix(np.array(M))
    K = sparse.csc_matrix(np.array(K))
    C = sparse.csc_matrix(np.array(C))
    F = sparse.csc_matrix(np.array(F))

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2
    return M, K, C, F, n_steps, time, number_eq


@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_central_difference_sparse(setup_module, linear_solver, preconditioner):
    """
    Test Central Difference solver with different linear solvers and preconditioners using sparse matrices.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = CentralDifferenceSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=prec,
                                  lumping_method=LumpingMethod.RowSum)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [0.000, 0.392],
                    [0.0307, 1.45],
                    [0.168, 2.83],
                    [0.487, 4.14],
                    [1.02, 5.02],
                    [1.7, 5.26],
                    [2.4, 4.9],
                    [2.91, 4.17],
                    [3.07, 3.37],
                    [2.77, 2.78],
                    [2.04, 2.54],
                    [1.02, 2.60],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_central_difference_sparse_output_int(setup_module, linear_solver, preconditioner):
    """
    Test Central Difference solver with different linear solvers and preconditioners using sparse matrices.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = CentralDifferenceSolver(Force(), State(output_interval=10), linear_solver=linear_solver(), preconditioner=prec,
    lumping_method=LumpingMethod.RowSum)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [2.77, 2.78],
                    [1.02, 2.60],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
def test_central_difference_dense(setup_module, linear_solver):
    """
    Test Central Difference solver with different linear solvers using dense matrices.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_np_array(M, K, C, F)

    res = CentralDifferenceSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=None,
                                  lumping_method=LumpingMethod.RowSum)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [0.000, 0.392],
                    [0.0307, 1.45],
                    [0.168, 2.83],
                    [0.487, 4.14],
                    [1.02, 5.02],
                    [1.7, 5.26],
                    [2.4, 4.9],
                    [2.91, 4.17],
                    [3.07, 3.37],
                    [2.77, 2.78],
                    [2.04, 2.54],
                    [1.02, 2.60],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_central_difference_static(setup_module, linear_solver, preconditioner):
    """
    Test Central Difference solver with a lot of damping to see if solution converges to static solution
    """

    M, K, _, F, _, _, number_eq = setup_module

    n_steps = 500
    t_step = 0.28
    t_total = n_steps * t_step
    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step)))

    F = sparse.csc_matrix(np.zeros((2, 500)))
    F[1, :] = 10

    # rayleigh damping matrix
    f1 = 1
    f2 = 10
    d1 = 1
    d2 = 1
    damp_mat = (
        1 / 2
        * np.array([[1 / (2 * np.pi * f1), 2 * np.pi * f1],
                    [1 / (2 * np.pi * f2), 2 * np.pi * f2],
                    ]
                    )
                )
    damp_qsi = np.array([d1, d2])
    # solution
    alpha, beta = np.linalg.solve(damp_mat, damp_qsi)
    damp = M.dot(alpha) + K.dot(beta)

    prec = preconditioner() if preconditioner is not None else None

    res = CentralDifferenceSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, damp, K, F, 0, n_steps -1)

    # check solution
    np.testing.assert_array_almost_equal(np.round(res.u[0], 2), np.round(np.array([0, 0]), 2))
    np.testing.assert_array_almost_equal(np.round(res.u[-1], 2), np.round(np.array([1, 3]), 2))
