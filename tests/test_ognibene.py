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

        n_time_steps =1000

        duration = 1
        t = np.linspace(0, duration, n_time_steps+1)
        # x = np.linspace(0, 6 * np.pi, n_time_steps+1)


        freq =3 # load frequency [hz]
        # Scale and shift to get values between 5 and 100
        max_value = 98e3
        min_value = 5e3
        amplitude = (max_value - min_value) / 2
        offset = (max_value + min_value) / 2
        # offset = min_value
        # scaled_sine = amplitude * basic_sine + offset
        scaled_sine = amplitude * np.cos(2 * np.pi * freq * t  + np.pi ) + offset
        # f = np.sin()
        f = scaled_sine

        n_equations = 3

        self.M = np.zeros((n_equations,n_equations))
        self.K = np.zeros((n_equations,n_equations))
        self.C = np.zeros((n_equations,n_equations))
        self.F = np.zeros((n_equations,n_time_steps+1))

        # rail
        mass_rail = 60
        M_rail = np.array([[mass_rail]])
        equation_nbrs_rail = [0]
        self.M[equation_nbrs_rail, equation_nbrs_rail] = self.M[equation_nbrs_rail, equation_nbrs_rail] + M_rail

        # railpad
        krp = 230e6
        crp = 75e3
        K_rp = np.array([[krp, -krp], [-krp, krp]])
        C_rp = np.array([[crp, -crp], [-crp, crp]])
        equation_nbrs_rp = np.array([0, 1])
        self.K[np.ix_(equation_nbrs_rp, equation_nbrs_rp)] = self.K[np.ix_(equation_nbrs_rp, equation_nbrs_rp)] + K_rp
        self.C[np.ix_(equation_nbrs_rp, equation_nbrs_rp)] = self.C[np.ix_(equation_nbrs_rp, equation_nbrs_rp)] + C_rp


        # sleeper
        mass_slp = 160
        M_slp = np.array([[mass_slp]])
        equation_nbrs_slp = [1]
        self.M[np.ix_(equation_nbrs_slp, equation_nbrs_slp)] = self.M[np.ix_(equation_nbrs_slp, equation_nbrs_slp)] + M_slp

        # ballast
        kb = 210e6
        cb = 0.12e6
        K_b = np.array([[kb, -kb], [-kb, kb]])
        C_b = np.array([[cb, -cb], [-cb, cb]])
        equation_nbrs_b = [1, 2]
        self.K[np.ix_(equation_nbrs_b, equation_nbrs_b)] = self.K[np.ix_(equation_nbrs_b, equation_nbrs_b)] + K_b
        self.C[np.ix_(equation_nbrs_b, equation_nbrs_b)] = self.C[np.ix_(equation_nbrs_b, equation_nbrs_b)] + C_b

        #mass ballast
        mass_ballast = 300
        M_ballast = np.array([[mass_ballast]])
        equation_nbrs_ballast = [2]
        self.M[np.ix_(equation_nbrs_ballast, equation_nbrs_ballast)] = self.M[np.ix_(equation_nbrs_ballast, equation_nbrs_ballast)] + M_ballast

        # foundation
        kf = 750e6
        cf = 0.15e6
        K_f = np.array([[kf]])
        C_f = np.array([[cf]])
        equation_nbrs_f = [2]
        self.K[np.ix_(equation_nbrs_f, equation_nbrs_f)] = self.K[np.ix_(equation_nbrs_f, equation_nbrs_f)] + K_f
        self.C[np.ix_(equation_nbrs_f, equation_nbrs_f)] = self.C[np.ix_(equation_nbrs_f, equation_nbrs_f)] + C_f


        # self.M = np.array([[300]])
        # self.K = np.array([[kb]])
        # self.C = np.array([[cb]])
        self.F[equation_nbrs_rail,:] = f[None,:]



        self.u0 = np.zeros(n_equations)
        self.v0 = np.zeros(n_equations)

        self.n_steps = n_time_steps

        self.time = t

        # self.t_step = 0.1
        # self.t_total = self.n_steps * self.t_step
        self.t_total = duration
        self.t_step = self.t_total / self.n_steps

        # self.time = np.linspace(
        #     0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step)+1)
        # )

        self.number_eq = n_equations
        return



    def run_newmark_test(self,solver):
        res = solver()

        res.beta = self.settings["beta"]
        res.gamma = self.settings["gamma"]

        res.initialise(self.number_eq, self.time)

        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps, 1000)
        #
        # plt.plot(self.time, res.u[:, 0])
        # plt.plot(self.time, res.u[:, 1])
        # plt.plot(self.time, res.u[:, 2])
        # plt.show()
        # plt.show()
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
