import pytest

import numpy as np
from scipy import sparse

from solvers.base_solver import Force, State
from solvers.HHT_solver import HHTExplicit, HHTImplicitForce
from solvers.base_solver import Force, State
from solvers.linear_equations_solvers import (SparseDirectSolver, SparseDirectSolverLU, DenseDirectSolver,
    CGSolver, GMRESSolver, BICSTABSolver)
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner

from tests.utils import *


@pytest.fixture
def setup_module():
    """
    Setup matrices for testing HHT solver.
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



@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverLU, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("hht", [HHTExplicit, HHTImplicitForce])
def test_hht_sparse(setup_module, linear_solver, preconditioner, hht):
    """
    Test HHT solver with sparse matrices.
    Results are compared to reference solution from Bathe for Newmark. HHT reduced to Newmark when alpha=0.

    Args:
        setup_module: fixture that sets up matrices and parameters
        linear_solver: linear solver class to use
        preconditioner: preconditioner class to use
        hht: HHT solver class to test
    """

    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = hht(Force(), State(), linear_solver=linear_solver(), preconditioner=prec, alpha=0)
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

@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverLU,CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("hht", [HHTExplicit, HHTImplicitForce])
def test_hht_sparse_output_int(setup_module, linear_solver, preconditioner, hht):
    """
    Test HHT solver with sparse matrices.
    Results are compared to reference solution from Bathe for Newmark. HHT reduced to Newmark when alpha=0.

    Args:
        setup_module: fixture that sets up matrices and parameters
        linear_solver: linear solver class to use
        preconditioner: preconditioner class to use
        hht: HHT solver class to test
    """

    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = hht(Force(), State(output_interval=10), linear_solver=linear_solver(), preconditioner=prec, alpha=0)
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


@pytest.mark.parametrize("linear_solver", [DenseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("hht", [HHTExplicit, HHTImplicitForce])
def test_hht_dense(setup_module, linear_solver, hht):
    """
    Test HHT solver with different linear solvers using dense matrices.
    Results are compared to reference solution from Bathe for Newmark. HHT reduced to Newmark when alpha=0.

    Args:
        setup_module: Fixture setting up matrices
        linear_solver: Linear solver class
        hht: HHT solver class
    """
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_np_array(M, K, C, F)

    res = hht(Force(), State(), linear_solver=linear_solver(), preconditioner=None, alpha=0)
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


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverLU, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
@pytest.mark.parametrize("hht", [HHTExplicit, HHTImplicitForce])
def test_hht_two_stages(setup_module, linear_solver, preconditioner, hht):
    """
    Test HHT solver with 2 stages, where the different stages have different time steps.
    Results are compared to reference solution from Bathe for Newmark. HHT reduced to Newmark when alpha=0.

    Args:
        setup_module: fixture that sets up matrices and parameters
        linear_solver: linear solver class to use
        preconditioner: preconditioner class to use
        hht: HHT solver class to test
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
    res = hht(Force(), State(), linear_solver=linear_solver(), preconditioner=prec, alpha=0)
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
