import pytest
import numpy as np
from scipy import sparse

from solvers.central_difference_solver import CentralDifferenceSolver
from solvers.utils import LumpingMethod
from tests.utils import set_matrices_as_sparse, set_matrices_as_np_array


@pytest.fixture
def central_difference_basic_setup():
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
    # example from bathe
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
    # example from bathe
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


def run_central_difference_test(setup_data, lumped):
    """
    Helper function to run the central difference solver test with given parameters
    """
    if lumped:
        lump = LumpingMethod.RowSum
    else:
        lump = LumpingMethod.NONE
    res = CentralDifferenceSolver(lumping_method=lump)

    res.initialise(setup_data['number_eq'], setup_data['time'])
    res.calculate(setup_data['M'], setup_data['C'], setup_data['K'],
                 setup_data['F'], 0, setup_data['n_steps'])

    # check solution
    expected_results = np.array(
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
    )
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(expected_results, 2)
    )


def run_central_difference_test_damping(setup_data, lumped):
    """
    Helper function to run the central difference solver test with damping
    """
    if lumped:
        lump = LumpingMethod.RowSum
    else:
        lump = LumpingMethod.NONE
    res = CentralDifferenceSolver(lumping_method=lump)

    res.initialise(setup_data['number_eq'], setup_data['time'])
    res.calculate(setup_data['M'], setup_data['C'], setup_data['K'],
                 setup_data['F'], 0, setup_data['n_steps'])

    # check solution
    expected_results = np.array(
        [
            [0, 0],
            [0, 0.392],
            [0.0299, 1.3684],
            [0.1557, 2.5818],
            [0.4359, 3.6653],
            [0.8807, 4.3525],
            [1.4316, 4.5475],
            [1.9719, 4.3263],
            [2.3615, 3.879 ],
            [2.4854, 3.4203],
            [2.2948, 3.106 ],
            [1.8264, 2.9857],
            [1.1933, 3.0052]
        ]
    )
    np.testing.assert_array_almost_equal(
        np.round(res.u, 2),
        np.round(expected_results, 2)
    )


def test_full_solver_central_difference(central_difference_basic_setup):
    setup_data = central_difference_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test(setup_data, lumped=False)


def test_full_solver_central_difference_lumped(central_difference_basic_setup):
    setup_data = central_difference_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test(setup_data, lumped=True)


def test_sparse_solver_central_difference(central_difference_basic_setup):
    setup_data = central_difference_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test(setup_data, lumped=False)


def test_sparse_solver_central_difference_lumped(central_difference_basic_setup):
    setup_data = central_difference_basic_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test(setup_data, lumped=True)


# Full matrix tests
def test_full_solver_central_difference_consistent_lumped(full_matrix_setup):
    setup_data = full_matrix_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test(setup_data, lumped=True)


def test_sparse_solver_central_difference_consistent_lumped(full_matrix_setup):
    setup_data = full_matrix_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test(setup_data, lumped=True)


def test_full_solver_central_difference_full_damp_lumped(full_matrix_damping_setup):
    setup_data = full_matrix_damping_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_np_array(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test_damping(setup_data, lumped=True)


def test_sparse_solver_central_difference_consistent_damp_lump(full_matrix_damping_setup):
    setup_data = full_matrix_damping_setup.copy()
    setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F'] = set_matrices_as_sparse(
        setup_data['M'], setup_data['K'], setup_data['C'], setup_data['F']
    )
    run_central_difference_test_damping(setup_data, lumped=True)

