import pytest
import numpy as np
from scipy import sparse

from solvers.bathe_solver import BatheSolver
from tests.utils import set_matrices_as_sparse, set_matrices_as_np_array


@pytest.fixture
def bathe_basic_setup():
    # example from bathe for CDM
    M = [[2, 0], [0, 1]]
    K = [[6, -2], [-2, 4]]
    C = [[0, 0], [0, 0]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M_mat = sparse.csc_matrix(np.array(M))
    K_mat = sparse.csc_matrix(np.array(K))
    C_mat = sparse.csc_matrix(np.array(C))
    F_mat = sparse.csc_matrix(np.array(F))

    u0 = np.zeros(2)
    v0 = np.zeros(2)

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2

    return {
        'M': M_mat, 'K': K_mat, 'C': C_mat, 'F': F_mat,
        'u0': u0, 'v0': v0, 'n_steps': n_steps, 't_step': t_step,
        't_total': t_total, 'time': time, 'number_eq': number_eq
    }


@pytest.fixture
def full_matrix_setup():
    M = [[1, 1], [0.25, 0.75]]
    K = [[6, -2], [-2, 4]]
    C = [[0, 0], [0, 0]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M_mat = sparse.csc_matrix(np.array(M))
    K_mat = sparse.csc_matrix(np.array(K))
    C_mat = sparse.csc_matrix(np.array(C))
    F_mat = sparse.csc_matrix(np.array(F))

    u0 = np.zeros(2)
    v0 = np.zeros(2)

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2

    return {
        'M': M_mat, 'K': K_mat, 'C': C_mat, 'F': F_mat,
        'u0': u0, 'v0': v0, 'n_steps': n_steps, 't_step': t_step,
        't_total': t_total, 'time': time, 'number_eq': number_eq
    }


@pytest.fixture
def full_matrix_damping_setup():
    M = [[1, 1], [0.25, 0.75]]
    K = [[6, -2], [-2, 4]]
    C = [[0.25, 0.15], [0.15, 0.25]]
    F = np.zeros((2, 13))
    F[1, :] = 10
    M_mat = sparse.csc_matrix(np.array(M))
    K_mat = sparse.csc_matrix(np.array(K))
    C_mat = sparse.csc_matrix(np.array(C))
    F_mat = sparse.csc_matrix(np.array(F))

    u0 = np.zeros(2)
    v0 = np.zeros(2)

    n_steps = 12
    t_step = 0.28
    t_total = n_steps * t_step

    time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / t_step) + 1))

    number_eq = 2

    return {
        'M': M_mat, 'K': K_mat, 'C': C_mat, 'F': F_mat,
        'u0': u0, 'v0': v0, 'n_steps': n_steps, 't_step': t_step,
        't_total': t_total, 'time': time, 'number_eq': number_eq
    }


def run_bathe_test(setup_data, solver, lumped):
    """
    Helper function to run the bathe solver test with given parameters
    """
    res = solver(lumped=lumped)

    res.initialise(setup_data['number_eq'], setup_data['time'])
    res.calculate(setup_data['M'], setup_data['C'], setup_data['K'],
                 setup_data['F'], 0, setup_data['n_steps'])

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


def run_bathe_test_damping(setup_data, solver, lumped):
    """
    Helper function to run the bathe solver test with damping
    """
    res = solver(lumped=lumped)

    res.initialise(setup_data['number_eq'], setup_data['time'])
    res.calculate(setup_data['M'], setup_data['C'], setup_data['K'],
                 setup_data['F'], 0, setup_data['n_steps'])

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


def test_solver_bathe_consistent_matrix(bathe_basic_setup):
    setup_data = bathe_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test(setup_data, BatheSolver, lumped=False)


def test_solver_bathe_lumped_matrix(bathe_basic_setup):
    setup_data = bathe_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test(setup_data, BatheSolver, lumped=True)


def test_solver_bathe_consistent_matrix_sparse(bathe_basic_setup):
    setup_data = bathe_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test(setup_data, BatheSolver, lumped=False)


def test_solver_bathe_lumped_matrix_sparse(bathe_basic_setup):
    setup_data = bathe_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test(setup_data, BatheSolver, lumped=True)


def test_solver_bathe_lumped_matrix_full(full_matrix_setup):
    setup_data = full_matrix_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test(setup_data, BatheSolver, lumped=True)


def test_solver_bathe_lumped_matrix_full_sparse(full_matrix_setup):
    setup_data = full_matrix_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test(setup_data, BatheSolver, lumped=True)


def test_solver_bathe_lumped_matrix_full_damping(full_matrix_damping_setup):
    setup_data = full_matrix_damping_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test_damping(setup_data, BatheSolver, lumped=True)


def test_solver_bathe_lumped_matrix_full_sparse_damping(full_matrix_damping_setup):
    setup_data = full_matrix_damping_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_bathe_test_damping(setup_data, BatheSolver, lumped=True)

