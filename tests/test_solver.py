import pytest

import numpy as np
from scipy import sparse

from solvers.base_solver import Force, State, calculate_initial_acceleration, TimeIntegrationType
from solvers.linear_equations_solvers import SparseDirectSolver
from solvers.newmark_solver import NewmarkExplicit, NewmarkImplicitForce
from solvers.central_difference_solver import CentralDifferenceSolver
from solvers.bathe_solver import BatheSolver
from solvers.HHT_solver import HHTImplicitForce, HHTExplicit
from solvers.zhai_solver import ZhaiSolver
from solvers.static_solver import StaticSolver

from tests.utils import set_matrices_as_sparse, ALL_DYNAMIC_SOLVERS, ALL_SPARSE_LINEAR_SOLVERS, ALL_PRECONDITIONERS


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

    K = sparse.csc_matrix(np.array(K))
    F = sparse.csc_matrix(np.array(F))

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(
        0, t_total, int(np.ceil((t_total - 0) / t_step) + 1)
    )

    number_eq = 2
    return M, K, C, F, n_steps, time, number_eq


def test_force_time_exception(setup_module):
    """
    Test if ValueError is raised when time vector of solver and force vector do not match.

    Args:
        setup_module: Fixture to setup the module
    """

    _, K, _, F, n_steps, time, number_eq = setup_module
    res = StaticSolver(Force(), State(), SparseDirectSolver())

    # redefine time with different length
    time = np.linspace(0, 1, 5)

    res.initialise(number_eq, time)

    # check if ValueError is raised for time mismatch
    with pytest.raises(ValueError, match="Solver time is not equal to force vector time"):
        res.calculate(K, F, 0, n_steps - 1)


@pytest.mark.parametrize("solver", ALL_DYNAMIC_SOLVERS)
def test_load_function(setup_module, solver):
    """
    Test if Newmark solver returns equal results while using a load function and an initial force matrix. The load
    function is chosen such that at each time step it calculates the same value as the tested force matrix at that
    time step.

    Args:
        setup_module: Fixture to setup the module
    """

    M, K, C, F, n_steps, time, number_eq = setup_module

    def load_function(t, u=None):
        # half load each time step
        if t>0:
            F[:, t] = F[:,t-1] * 0.5
        return F[:, t].toarray()[:,0]

    # manually make force matrix
    force_matrix = np.zeros(F.shape)
    force_matrix[:, 0] = F.toarray()[:,0]
    for i in range(1, force_matrix.shape[1]):
        force_matrix[:, i] = force_matrix[:, i-1]/2

    # calculate using custom load function
    res_func = solver(Force(), State())
    res_func.force.update_rhs_at_time_step_func = load_function
    res_func.initialise(number_eq, time)
    res_func.calculate(M, C, K, F, 0, n_steps)

    # calculate using initial load matrix
    res_manual = solver(Force(), State())
    res_manual.initialise(number_eq, time)
    res_manual.calculate(M, C, K, force_matrix, 0, n_steps)

    # check if solutions are equal
    np.testing.assert_array_almost_equal(res_func.u, res_manual.u)


@pytest.mark.parametrize("linear_solver", ALL_SPARSE_LINEAR_SOLVERS)
@pytest.mark.parametrize("preconditioner", ALL_PRECONDITIONERS)
def test_initial_acceleration(setup_module, linear_solver, preconditioner):

    M, K, C, F, _, _, number_eq = setup_module
    M, K, C, F = set_matrices_as_sparse(M, K, C, F)

    prec = preconditioner() if preconditioner is not None else None

    acc = calculate_initial_acceleration(M, C, K, F[:, 0].toarray()[:, 0],
                                        np.zeros(number_eq), np.zeros(number_eq),
                                        linear_solver(), prec)
    np.testing.assert_array_equal(acc, np.array([0, 10]))


@pytest.mark.parametrize("solver_cls, expected_type",[(StaticSolver, TimeIntegrationType.STATIC),
                                                      (NewmarkExplicit, TimeIntegrationType.DYNAMIC),
                                                      (NewmarkImplicitForce, TimeIntegrationType.DYNAMIC),
                                                      (CentralDifferenceSolver, TimeIntegrationType.DYNAMIC),
                                                      (BatheSolver, TimeIntegrationType.DYNAMIC),
                                                      (HHTImplicitForce, TimeIntegrationType.DYNAMIC),
                                                      (HHTExplicit, TimeIntegrationType.DYNAMIC),
                                                      (ZhaiSolver, TimeIntegrationType.DYNAMIC),
                                                      ],
                                                      )
def test_solver_time_integration_types(solver_cls, expected_type):
    solver = solver_cls(Force(), State())
    assert solver.type is expected_type


