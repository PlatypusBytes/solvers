import pytest

import numpy as np
from scipy import sparse

from solvers.base_solver import Force, State
from solvers.linear_equations_solvers import (SparseDirectSolver, SparseDirectSolverLU, DenseDirectSolver,
    CGSolver, GMRESSolver, BICSTABSolver)
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner
from solvers.newmark_solver import NewmarkExplicit
from solvers.zhai_solver import ZhaiSolver

from tests.utils import *


@pytest.fixture
def setup_module():

    # edited example from bathe
    n_steps = 12 * 20
    t_step = 0.28 / 20
    t_total = n_steps * t_step

    M = [[2, 0], [0, 1]]
    K = [[6, -2], [-2, 4]]
    C = [[0, 0], [0, 0]]
    F = np.zeros((2, n_steps + 1))
    F[1, :] = 10
    M = sparse.csc_matrix(np.array(M))
    K = sparse.csc_matrix(np.array(K))
    C = sparse.csc_matrix(np.array(C))
    F = sparse.csc_matrix(np.array(F))

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2
    return M, K, C, F, n_steps, time, number_eq


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverLU, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
def test_zhai_sparse(setup_module, linear_solver, preconditioner):
    """
    Check if results following Zhai calculation are close to Newmark results for sparse matrices.

    Args:
        setup_module: Fixture to setup the module
        linear_solver: Linear solver to use
        preconditioner: Preconditioner to use
    """

    # run Newmark solver
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = NewmarkExplicit(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)


    # run Zhai solver
    res_2 = ZhaiSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res_2.initialise(number_eq, time)
    res_2.calculate(M, C, K, F, 0, n_steps)

    # assert
    np.testing.assert_array_almost_equal(np.round(res.u, 2), np.round(res_2.u, 2))


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverLU, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
def test_zhai_sparse_output_int(setup_module, linear_solver, preconditioner):
    """
    Check if results following Zhai calculation are close to Newmark results for sparse matrices.

    Args:
        setup_module: Fixture to setup the module
        linear_solver: Linear solver to use
        preconditioner: Preconditioner to use
    """

    # run Newmark solver
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = NewmarkExplicit(Force(), State(output_interval=10), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)


    # run Zhai solver
    res_2 = ZhaiSolver(Force(), State(output_interval=10), linear_solver=linear_solver(), preconditioner=prec)
    res_2.initialise(number_eq, time)
    res_2.calculate(M, C, K, F, 0, n_steps)

    # assert
    np.testing.assert_array_almost_equal(np.round(res.u, 2), np.round(res_2.u, 2))


@pytest.mark.parametrize("linear_solver", [DenseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver])
def test_zhai_dense(setup_module, linear_solver):
    """
    Check if results following Zhai calculation are close to Newmark results for np.array matrices.

    Args:
        setup_module: Fixture to setup the module
        linear_solver: Linear solver to use
    """

    # run Newmark solver
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_np_array(M, K, C, F)

    res = NewmarkExplicit(Force(), State(), linear_solver=linear_solver(), preconditioner=None)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps)


    # run Zhai solver
    res_2 = ZhaiSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=None)
    res_2.initialise(number_eq, time)
    res_2.calculate(M, C, K, F, 0, n_steps)

    # assert
    np.testing.assert_array_almost_equal(np.round(res.u, 2), np.round(res_2.u, 2))


@pytest.mark.parametrize("linear_solver", [SparseDirectSolver, SparseDirectSolverLU, CGSolver, GMRESSolver, BICSTABSolver])
@pytest.mark.parametrize("preconditioner", [None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner])
def test_zhai_sparse_two_stages(setup_module, linear_solver, preconditioner):
    """
    Check if results following Zhai calculation are close to Newmark results for sparse matrices for two stages.

    Args:
        setup_module: Fixture to setup the module
        linear_solver: Linear solver to use
        preconditioner: Preconditioner to use
    """

    # run Newmark solver
    M, K, C, F, n_steps, time, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    res = NewmarkExplicit(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res.initialise(number_eq, time)
    res.calculate(M, C, K, F, 0, n_steps // 2)
    res.state.update_initial_conditions(n_steps // 2)
    res.calculate(M, C, K, F, n_steps // 2, n_steps)


    # run Zhai solver
    res_2 = ZhaiSolver(Force(), State(), linear_solver=linear_solver(), preconditioner=prec)
    res_2.initialise(number_eq, time)
    res_2.calculate(M, C, K, F, 0, n_steps // 2)
    res_2.state.update_initial_conditions(n_steps // 2)
    res_2.calculate(M, C, K, F, n_steps // 2, n_steps)

    # assert
    np.testing.assert_array_almost_equal(np.round(res.u, 2), np.round(res_2.u, 2))
