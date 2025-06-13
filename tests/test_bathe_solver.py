import unittest

from solvers.bathe_solver import BatheSolver

from tests.utils import set_matrices_as_sparse, set_matrices_as_np_array

import numpy as np
from scipy import sparse

class TestBathe(unittest.TestCase):
    def setUp(self):

        # example from bathe
        M = [[2, 0], [0, 1]]
        K = [[6, -2], [-2, 4]]
        C = [[0, 0], [0, 0]]
        F = np.zeros((2, 13))
        F[1, :] = 10
        self.M = sparse.csc_matrix(np.array(M))
        self.K = sparse.csc_matrix(np.array(K))
        self.C = sparse.csc_matrix(np.array(C))
        self.F = sparse.csc_matrix(np.array(F))

        self.u0 = np.zeros(2)
        self.v0 = np.zeros(2)

        self.n_steps = 12
        self.t_step = 0.28
        self.t_total = self.n_steps * self.t_step

        self.time = np.linspace(
            0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step)+1)
        )

        self.number_eq = 2
        return

    def run_bathe_test(self, solver, lumped):
        res = solver(lumped=lumped)

        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        # check solution
        np.testing.assert_array_almost_equal(
            np.round(res.u, 2),
            np.round(
                np.array(
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
                ),
                2,
            ),
        )

    def test_nd_array_solver_central_difference(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=False)

    def test_nd_array_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=True)

    def test_sparse_solver_central_difference(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=False)

    def test_sparse_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=True)


class TestCentralDifferenceFull(unittest.TestCase):
    def setUp(self):

        # example from bathe
        M = [[1, 1], [0.25, 0.75]]
        K = [[6, -2], [-2, 4]]
        C = [[0, 0], [0, 0]]
        F = np.zeros((2, 13))
        F[1, :] = 10
        self.M = sparse.csc_matrix(np.array(M))
        self.K = sparse.csc_matrix(np.array(K))
        self.C = sparse.csc_matrix(np.array(C))
        self.F = sparse.csc_matrix(np.array(F))

        self.u0 = np.zeros(2)
        self.v0 = np.zeros(2)

        self.n_steps = 12
        self.t_step = 0.28
        self.t_total = self.n_steps * self.t_step

        self.time = np.linspace(
            0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step)+1)
        )

        self.number_eq = 2
        return

    def run_bathe_test(self, solver, lumped):
        res = solver(lumped=lumped)

        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        # check solution
        np.testing.assert_array_almost_equal(
            np.round(res.u, 2),
            np.round(
                np.array(
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
                ),
                2,
            ),
        )

    def test_nd_array_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=True)

    def test_sparse_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=True)


class TestCentralDifferenceFullDamping(unittest.TestCase):
    def setUp(self):

        # example from bathe
        M = [[1, 1], [0.25, 0.75]]
        K = [[6, -2], [-2, 4]]
        C = [[0.25, 0.15], [0.15, 0.25]]
        F = np.zeros((2, 13))
        F[1, :] = 10
        self.M = sparse.csc_matrix(np.array(M))
        self.K = sparse.csc_matrix(np.array(K))
        self.C = sparse.csc_matrix(np.array(C))
        self.F = sparse.csc_matrix(np.array(F))

        self.u0 = np.zeros(2)
        self.v0 = np.zeros(2)

        self.n_steps = 12
        self.t_step = 0.28
        self.t_total = self.n_steps * self.t_step

        self.time = np.linspace(
            0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step)+1)
        )

        self.number_eq = 2
        return

    def run_bathe_test(self, solver, lumped):
        res = solver(lumped=lumped)

        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        # check solution
        np.testing.assert_array_almost_equal(
            np.round(res.u, 2),
            np.round(
                np.array(
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
                ),
                2,
            ),
        )

    def test_nd_array_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=True)

    def test_sparse_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_bathe_test(BatheSolver, lumped=True)

