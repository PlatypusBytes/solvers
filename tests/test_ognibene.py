# unit test for solver
# tests based on Bathe
# for newmark pg 782
import unittest
import pytest

from solvers.newmark_ognibene import NewmarkOgniBene

from tests.utils import *

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt


class TestOgniBene(unittest.TestCase):
    def setUp(self):
        # newmark settings
        self.settings = {
            "beta": 0.25,
            "gamma": 0.5,
        }

        n_time_steps =100

        kb = 1
        cb = 1

        f = np.linspace(0.1, 10, n_time_steps+1)

        self.M = np.array([[0]])
        self.K = np.array([[kb]])
        self.C = np.array([[cb]])
        self.F = f[None,:]



        self.u0 = np.zeros(1)
        self.v0 = np.zeros(1)

        self.n_steps = n_time_steps
        self.t_step = 0.1
        self.t_total = self.n_steps * self.t_step

        self.time = np.linspace(
            0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step)+1)
        )

        self.number_eq = 1
        return



    def run_newmark_test(self,solver):
        res = solver()

        res.beta = self.settings["beta"]
        res.gamma = self.settings["gamma"]

        res.initialise(self.number_eq, self.time)

        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        plt.plot(self.time, res.u[:, 0])
        plt.show()
        a=1+1
        # check solution
        # np.testing.assert_array_almost_equal(
        #     np.round(res.u, 2),
        #     np.round(
        #         np.array(
        #             [
        #                 [0, 0],
        #                 [0.00673, 0.364],
        #                 [0.0505, 1.35],
        #                 [0.189, 2.68],
        #                 [0.485, 4.00],
        #                 [0.961, 4.95],
        #                 [1.58, 5.34],
        #                 [2.23, 5.13],
        #                 [2.76, 4.48],
        #                 [3.00, 3.64],
        #                 [2.85, 2.90],
        #                 [2.28, 2.44],
        #                 [1.40, 2.31],
        #             ]
        #         ),
        #         2,
        #     ),
        # )



    def test_np_array_solver_newmark_ognibene(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_newmark_test(NewmarkOgniBene)


        a=1+1
