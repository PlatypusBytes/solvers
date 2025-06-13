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
                        [0, 0],
                        [0.002, 0.38],
                        [0.037, 1.42],
                        [0.17, 2.79],
                        [0.49, 4.10],
                        [1.00, 5.],
                        [1.66, 5.28],
                        [2.35, 4.97],
                        [2.87, 4.26],
                        [3.05, 3.45],
                        [2.80, 2.81],
                        [2.12, 2.49],
                        [1.14, 2.51],
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
                        [0, 0],
                        [0.002, 0.38],
                        [0.037, 1.42],
                        [0.17, 2.79],
                        [0.49, 4.10],
                        [1.00, 5.],
                        [1.66, 5.28],
                        [2.35, 4.97],
                        [2.87, 4.26],
                        [3.05, 3.45],
                        [2.80, 2.81],
                        [2.12, 2.49],
                        [1.14, 2.51],
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
                        [0, 0],
                        [0, 0.38],
                        [0.02, 1.36],
                        [0.11, 2.61],
                        [0.35, 3.75],
                        [0.76, 4.48],
                        [1.31, 4.68],
                        [1.90, 4.38],
                        [2.37, 3.81],
                        [2.59, 3.23],
                        [2.47, 2.85],
                        [2.01, 2.76],
                        [1.33, 2.93],
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

