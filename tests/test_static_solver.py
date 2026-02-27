import pytest

import numpy as np
from scipy import sparse

from solvers.static_solver import StaticSolver
from solvers.base_solver import Force, State
from tests.utils import ALL_LINEAR_SOLVERS, ALL_PRECONDITIONERS, ALL_DENSE_LINEAR_SOLVERS


@pytest.fixture
def setup_module():
    """
    Setup matrices for testing newmark solver.
    Example from Bathe Chp: 9.2.4 (pg 794).

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
    return K, F, n_steps, time, number_eq



@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_solver_static_sparse(setup_module, linear_solver, preconditioner):
    """
    Static solver test with sparse matrices

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
    """
    K, F, n_steps, time, number_eq = setup_module

    K = sparse.csc_matrix(K)
    F = sparse.csc_matrix(F)

    prec = preconditioner() if preconditioner is not None else None

    res = StaticSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(K, F, 0, n_steps)
    # check static solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_solver_static_sparse_output_int(setup_module, linear_solver, preconditioner):
    """
    Static solver test with sparse matrices

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
    """
    K, F, n_steps, time, number_eq = setup_module

    K = sparse.csc_matrix(K)
    F = sparse.csc_matrix(F)

    prec = preconditioner() if preconditioner is not None else None

    res = StaticSolver(Force(), State(output_interval=10), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(K, F, 0, n_steps)
    # check static solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", ALL_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_solver_static_sparse_output_int_staged(setup_module, linear_solver, preconditioner):
    """
    Static solver test with sparse matrices

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
    """
    K, F, n_steps, time, number_eq = setup_module

    K = sparse.csc_matrix(K)
    F = sparse.csc_matrix(F)

    prec = preconditioner() if preconditioner is not None else None

    res = StaticSolver(Force(), State(output_interval=1), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(K, F, 0, n_steps)
    res.state.update_initial_conditions(n_steps // 2)
    res.calculate(K, F, n_steps // 2, n_steps, F_ini=F[:, n_steps // 2])
    # check static solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", ALL_DENSE_LINEAR_SOLVERS)
def test_solver_static_np_array(setup_module, linear_solver):
    """
    Static solver test with numpy array matrices

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
    """
    K, F, n_steps, time, number_eq = setup_module

    K = K.toarray()
    F = F.toarray()

    res = StaticSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=None)
    res.initialise(number_eq, time)
    res.calculate(K, F, 0, n_steps)
    # check static solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                ]
            ),
            2,
        ),
    )
