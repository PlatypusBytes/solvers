# unit test for solver
# tests based on Bathe
# for newmark pg 782
import unittest
import pytest

from solvers.HHT_solver import HHTExplicit

from tests.utils import *

import numpy as np
from scipy import sparse


class TestHHT(unittest.TestCase):
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

    def run_hht_test(self,solver):
        res = solver()

        res.initialise(self.number_eq, self.time)

        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        # check solution
        np.testing.assert_array_almost_equal(
            np.round(res.u, 2),
            np.round(
                np.array(
                    [
                        [0, 0],
                        [0.00673, 0.364],
                        [0.0505, 1.35],
                        [0.189, 2.68],
                        [0.485, 4.00],
                        [0.961, 4.95],
                        [1.58, 5.34],
                        [2.23, 5.13],
                        [2.76, 4.48],
                        [3.00, 3.64],
                        [2.85, 2.90],
                        [2.28, 2.44],
                        [1.40, 2.31],
                    ]
                ),
                2,
            ),
        )

    @pytest.mark.skip("work in progress")
    def test_sparse_solver_hht_explicit(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        # set_matrices_as_sparse()
        self.run_hht_test(HHTExplicit)
