# unit test for solver
# tests based on Bathe
# for newmark pg 782
import unittest
from solvers.static_solver import StaticSolver

from tests.utils import *

import numpy as np
from scipy import sparse

class TestStatic(unittest.TestCase):
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
        self.K = sparse.csc_matrix(np.array(K))
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


    def run_test_solver_static(self):
        res = StaticSolver()
        res.initialise(self.number_eq, self.time)
        res.calculate(self.K, self.F, 0, self.n_steps)
        # check static solution
        np.testing.assert_array_almost_equal(
            np.round(res.u, 2),
            np.round(
                np.array(
                    [
                        [0, 0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                        [1.0, 3.0],
                    ]
                ),
                2,
            ),
        )
        return


    def test_sparse_solver_static(self):
        _,self.K, _, self.F = set_matrices_as_sparse(None, self.K, None, self.F)
        self.run_test_solver_static()


    def test_np_array_solver_static(self):
        _, self.K, _, self.F = set_matrices_as_np_array(None, self.K, None, self.F)
        self.run_test_solver_static()

    def test_output_interval_static(self):
        _, self.K, _, self.F = set_matrices_as_np_array(None, self.K, None, self.F)

        output_interval = 10

        # write all output
        res = StaticSolver()
        res.initialise(self.number_eq, self.time)
        res.calculate(self.K, self.F, 0, self.n_steps)
        expected_displacement = np.concatenate((res.u[0::output_interval, :], res.u[None, -1, :]), axis=0)

        # write every other step
        res_2 = StaticSolver()
        res_2.output_interval = output_interval
        res_2.initialise(self.number_eq, self.time)
        res_2.calculate(self.K, self.F, 0, self.n_steps)

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, res_2.u)