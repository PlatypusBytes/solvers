import unittest

from solvers.central_difference_solver import CentralDifferenceSolver

from tests.utils import set_matrices_as_sparse, set_matrices_as_np_array

import numpy as np
from scipy import sparse

class TestCentralDifference(unittest.TestCase):
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

    def run_central_difference_test(self, solver, lumped):
        res = solver(lumped=lumped)

        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
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

    def test_nd_array_solver_central_difference(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=False)

    def test_nd_array_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=True)

    def test_sparse_solver_central_difference(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=False)

    def test_sparse_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=True)


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

    def run_central_difference_test(self, solver, lumped):
        res = solver(lumped=lumped)

        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
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

    def test_nd_array_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=True)

    def test_sparse_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=True)


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

    def run_central_difference_test(self, solver, lumped):
        res = solver(lumped=lumped)

        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        # check solution
        np.testing.assert_array_almost_equal(
            np.round(res.u, 2),
            np.round(
                np.array(
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
                ),
                2,
            ),
        )

    def test_nd_array_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=True)

    def test_sparse_solver_central_difference_lump(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_central_difference_test(CentralDifferenceSolver, lumped=True)

