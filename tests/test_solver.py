# unit test for solver
# tests based on Bathe
# for newmark pg 782
import unittest
from solvers.base_solver import Solver, TimeException
from solvers.newmark_solver import NewmarkSolver
from solvers.zhai_solver import ZhaiSolver
from solvers.static_solver import StaticSolver

import numpy as np
from scipy import sparse

class TestBaseSolver(unittest.TestCase):
    def setUp(self):
        K = [[6, -2], [-2, 4]]
        F = np.zeros((2, 13))
        F[1, :] = 10

        self.K = sparse.csc_matrix(np.array(K))
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

    def test_time_input_exception(self):
        res = StaticSolver()
        n_steps = 500
        t_total = n_steps * self.t_step
        time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / self.t_step)))
        res.initialise(self.number_eq, time)

        with self.assertRaises(TimeException) as exception:
            res.calculate(self.K, self.F, 0, n_steps - 1)

        self.assertTrue(
            "Solver time is not equal to force vector time" in exception.exception.args
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
