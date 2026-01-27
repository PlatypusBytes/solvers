import pytest
import numpy as np
from scipy import sparse

from solvers.base_solver import Force, State
from solvers.linear_equations_solvers import SparseDirectSolver, DenseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner
from solvers.newmark_solver import NewmarkExplicit, NewmarkImplicitForce

from tests.utils import *


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
    return M, K, C, F, n_steps, time, number_eq


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("newmark", [NewmarkExplicit, NewmarkImplicitForce])
def test_newmark_sparse(setup_module, linear_solver, preconditioner, newmark):
    """
    Test Newmark solver with different linear solvers and preconditioners using sparse matrices.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
        newmark: Newmark solver class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = newmark(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [0.00673, 0.364],
                    [0.0505, 1.35],
                    [0.189, 2.68],
                    [0.485, 4.00],
                    [0.961, 4.95],
                    [1.58, 5.34],
                    [2.23, 5.13],
                    [2.76, 4.48],
                    [3.00, 3.64],
                    [2.85, 2.90],
                    [2.28, 2.44],
                    [1.40, 2.31],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("newmark", [NewmarkExplicit, NewmarkImplicitForce])
def test_newmark_sparse_output_int(setup_module, linear_solver, preconditioner, newmark):
    """
    Test Newmark solver with different linear solvers and preconditioners using sparse matrices.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
        newmark: Newmark solver class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = newmark(Force(), State(output_interval=10), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [2.85, 2.90],
                    [1.40, 2.31],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("newmark", [NewmarkExplicit, NewmarkImplicitForce])
def test_newmark_sparse_output_int_staged(setup_module, linear_solver, preconditioner, newmark):
    """
    Test Newmark solver with different linear solvers and preconditioners using sparse matrices.
    The time to save the results does not align with the time steps, so the solver needs to add extra time steps.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        preconditioner: Preconditioner class
        newmark: Newmark solver class
    """
    M, K, C, F, _, time, number_eq = setup_module


    # set matrices as sparse
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    # run stages
    res = newmark(Force(), State(output_interval=4), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    # run first stage
    res.calculate(M, C, K, F, 0, 6)
    # run second stage
    res.state.update_initial_conditions(6)
    res.calculate(M, C, K, F, 6, len(time) - 1)

    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    # [0.00673, 0.364],
                    # [0.0505, 1.35],
                    # [0.189, 2.68],
                    [0.485, 4.00],
                    # [0.961, 4.95],
                    [1.58, 5.34],
                    # [2.23, 5.13],
                    [2.76, 4.48],
                    # [3.00, 3.64],
                    # [2.85, 2.90],
                    # [2.28, 2.44],
                    [1.40, 2.31],
                ]
            ),
            2,
        ),
    )


@pytest.mark.parametrize("linear_solver", [DenseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("newmark", [NewmarkExplicit, NewmarkImplicitForce])
def test_newmark_dense(setup_module, linear_solver, newmark):
    """
    Test Newmark solver with different linear solvers using dense matrices.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        newmark: Newmark solver class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_np_array(M, K, C, F)

    res = newmark(Force(), State(), linear_solver=linear_solver(), preconditioner=None)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)
    # check solution
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [0.00673, 0.364],
                    [0.0505, 1.35],
                    [0.189, 2.68],
                    [0.485, 4.00],
                    [0.961, 4.95],
                    [1.58, 5.34],
                    [2.23, 5.13],
                    [2.76, 4.48],
                    [3.00, 3.64],
                    [2.85, 2.90],
                    [2.28, 2.44],
                    [1.40, 2.31],
                ]
            ),
            2,
        ),
    )

@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("newmark", [NewmarkExplicit, NewmarkImplicitForce])
def test_newmark_two_stages(setup_module, linear_solver, preconditioner, newmark):
    """
    Test newmark solver with 2 stages, where the different stages have different time steps.

    Args:
        setup_module: fixture that sets up matrices and parameters
        linear_solver: linear solver class to use
        preconditioner: preconditioner class to use
        newmark: Newmark solver class to test
    """

    M, K, C, F, _, time, number_eq = setup_module

    # redefine time
    new_t_step = 0.5
    new_n_steps = 5
    new_t_start = time[-1] + new_t_step
    new_t_total = new_t_step * (new_n_steps - 1) + new_t_start
    time = np.concatenate((time, np.linspace(new_t_start, new_t_total, new_n_steps)))

    # redefine force vector
    F = np.zeros((2, len(time)))
    F[1, :] = 10

    # compute turning points
    diff = np.diff(time)
    turning_idxs = sorted(np.unique(diff.round(decimals=7), return_index=True)[1])

    # set matrices as sparse
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    # run stages
    res = newmark(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    # run first stage
    res.state.update_initial_conditions(turning_idxs[0])
    res.calculate(M, C, K, F, turning_idxs[0], turning_idxs[1])
    # run second stage
    res.state.update_initial_conditions(turning_idxs[1])
    res.calculate(M, C, K, F, turning_idxs[1], len(time) - 1)

    # check solution stage 1
    np.testing.assert_array_almost_equal(
        np.round(res.u[0:13, :], 2),
        np.round(
            np.array(
                [
                    [0, 0],
                    [0.00673, 0.364],
                    [0.0505, 1.35],
                    [0.189, 2.68],
                    [0.485, 4.00],
                    [0.961, 4.95],
                    [1.58, 5.34],
                    [2.23, 5.13],
                    [2.76, 4.48],
                    [3.00, 3.64],
                    [2.85, 2.90],
                    [2.28, 2.44],
                    [1.40, 2.31],
                ]
            ),
            2,
        ),
    )

    # check solution stage 2
    np.testing.assert_array_almost_equal(np.round(res.u[13:, :], 2), np.round(np.array([[-0.31, 2.56],
                                                                                        [-1.28, 2.70],
                                                                                        [-0.91, 2.31],
                                                                                        [0.52, 1.81],
                                                                                        [2.04, 2.08],
                                                                                        ]),
                                                                                        2,
                                                                                        ))


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("newmark", [NewmarkExplicit, NewmarkImplicitForce])
def test_newmark_static(setup_module, linear_solver, preconditioner, newmark):
    """
    Test newmark solver with a lot of damping to see if solution converges to static solution
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

    res = newmark(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, damp, K, F, 0, n_steps -1)

    # check solution
    np.testing.assert_array_almost_equal(np.round(res.u[0], 2), np.round(np.array([0, 0]), 2))
    np.testing.assert_array_almost_equal(np.round(res.u[-1], 2), np.round(np.array([1, 3]), 2))
