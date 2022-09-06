# unit test for solver
# tests based on Bathe
# for newmark pg 782
import unittest
from solvers.newmark_solver import NewmarkExplicit
from solvers.zhai_solver import ZhaiSolver

from tests.utils import *

import numpy as np
from scipy import sparse

class TestZhai(unittest.TestCase):
    def setUp(self):
        # newmark settings
        self.settings = {
            "beta": 0.25,
            "gamma": 0.5,
        }

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

        self.n_steps = 12 * 20
        self.t_step = 0.28 / 20
        self.t_total = self.n_steps * self.t_step

        self.time = np.linspace(
            0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step) + 1)
        )

        self.number_eq = 2
        return

    def run_test_zhai_solver(self):
        """
        Check if results following Zhai calculation are close to Newmark results
        :return:
        """

        # reshape force vector
        F = np.zeros((2, self.n_steps +1))
        F[1, :] = 10
        self.F = sparse.csc_matrix(np.array(F))

        # calculate with Newmark solver
        expected = NewmarkExplicit()
        expected.initialise(self.number_eq, self.time)
        expected.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # calculate with Zhai solver
        res = ZhaiSolver()
        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # assert
        np.testing.assert_array_almost_equal(np.round(expected.u, 2), np.round(res.u, 2))

    def test_sparse_solver_zhai(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        # set_matrices_as_sparse()
        self.run_test_zhai_solver()

    def test_np_array_solver_zhai(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_test_zhai_solver()

    def test_output_interval_zhai(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)

        # reshape force vector
        F = np.zeros((2, self.n_steps + 1))
        F[1, :] = 10
        self.F = sparse.csc_matrix(np.array(F))

        output_interval = 10

        # write all output
        res = ZhaiSolver()
        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        expected_displacement = res.u[0::output_interval,:]

        # write every other step
        res_2 = ZhaiSolver()
        res_2.output_interval = output_interval
        res_2.initialise(self.number_eq, self.time)
        res_2.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, res_2.u)
