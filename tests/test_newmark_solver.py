# unit test for solver
# tests based on Bathe
# for newmark pg 782
import unittest

from solvers.newmark_solver import NewmarkSolver

from tests.utils import *

import numpy as np
from scipy import sparse


class TestNewmark(unittest.TestCase):
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

        self.n_steps = 12
        self.t_step = 0.28
        self.t_total = self.n_steps * self.t_step

        self.time = np.linspace(
            0, self.t_total, int(np.ceil((self.t_total - 0) / self.t_step)+1)
        )

        self.number_eq = 2
        return

    def test_a_init(self):
        force = self.F[:, 0].toarray()[:, 0]
        # check computation of the acceleration
        solver = NewmarkSolver()

        solver._is_sparse_calculation = True
        acc = solver.calculate_initial_acceleration(self.M, self.C, self.K, force, self.u0, self.v0)

        # assert if true
        np.testing.assert_array_equal(acc, np.array([0, 10]))
        return

    def run_newmark_test(self):
        res = NewmarkSolver()

        res.beta = self.settings["beta"]
        res.gamma = self.settings["gamma"]

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

    def test_sparse_solver_newmark(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        # set_matrices_as_sparse()
        self.run_newmark_test()

    def test_np_array_solver_newmark(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_newmark_test()

    def run_test_solver_newmark_two_stages(self):
        """
               Test newmark solver with 2 stages, where the different stages have different time steps
               :return:
               """
        res = NewmarkSolver()

        res.beta = self.settings["beta"]
        res.gamma = self.settings["gamma"]

        new_t_step = 0.5
        new_n_steps = 5
        new_t_start = self.time[-1] + new_t_step
        new_t_total = new_t_step * (new_n_steps - 1) + new_t_start
        self.time = np.concatenate(
            (self.time, np.linspace(new_t_start, new_t_total, new_n_steps))
        )

        diff = np.diff(self.time)
        turning_idxs = sorted(np.unique(diff.round(decimals=7), return_index=True)[1])

        F = np.zeros((2, len(self.time)))
        F[1, :] = 10
        self.F = sparse.csc_matrix(np.array(F))

        res.initialise(self.number_eq, self.time)

        # run stages
        for i in range(len(turning_idxs) - 1):
            res.update(turning_idxs[i])
            res.calculate(
                self.M, self.C, self.K, self.F, turning_idxs[i], turning_idxs[i + 1]
            )
        res.update(turning_idxs[-1])
        res.calculate(
            self.M, self.C, self.K, self.F, turning_idxs[-1], len(self.time) - 1
        )

        # check solution stage 1
        np.testing.assert_array_almost_equal(
            np.round(res.u[0:13, :], 2),
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

        # check solution stage 2
        np.testing.assert_array_almost_equal(
            np.round(res.u[13:, :], 2),
            np.round(
                np.array([[-0.31, 2.56], [-1.28, 2.70], [-0.91, 2.31], [0.52, 1.81], [2.04, 2.08]]),
                2,
            ),
        )

    def test_np_array_solver_newmark_two_stages(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)
        self.run_test_solver_newmark_two_stages()

    def test_sparse_solver_newmark_two_stages(self):
        self.M, self.K, self.C, self.F = set_matrices_as_sparse(self.M, self.K, self.C, self.F)
        self.run_test_solver_newmark_two_stages()

    def test_solver_newmark_static(self):
        # with damping solution converges to the static one
        res = NewmarkSolver()

        n_steps = 500
        t_total = n_steps * self.t_step
        time = np.linspace(0, t_total, int(np.ceil((t_total - 0) / self.t_step)))
        res.initialise(self.number_eq, time)

        res.beta = self.settings["beta"]
        res.gamma = self.settings["gamma"]

        F = sparse.csc_matrix(np.zeros((2, 500)))
        F[1, :] = 10
        # rayleigh damping matrix
        f1 = 1
        f2 = 10
        d1 = 1
        d2 = 1
        damp_mat = (
            1
            / 2
            * np.array(
                [
                    [1 / (2 * np.pi * f1), 2 * np.pi * f1],
                    [1 / (2 * np.pi * f2), 2 * np.pi * f2],
                ]
            )
        )
        damp_qsi = np.array([d1, d2])
        # solution
        alpha, beta = np.linalg.solve(damp_mat, damp_qsi)
        damp = self.M.dot(alpha) + self.K.dot(beta)
        res.calculate(self.M, damp, self.K, F, 0, n_steps - 1)
        # check solution
        np.testing.assert_array_almost_equal(
            np.round(res.u[0], 2), np.round(np.array([0, 0]), 2)
        )
        np.testing.assert_array_almost_equal(
            np.round(res.u[-1], 2), np.round(np.array([1, 3]), 2)
        )
        return

    def test_load_function(self):
        """
        Test if Newmark solver returns equal results while using a load function and an initial force matrix. The load
        function is chosen such that at each time step it calculates the same value as the tested force matrix at that
        time step.
        :return:
        """

        def load_function(u,t):
            # half load each time step
            self.F[:, ] = self.F[:,t-1] * 0.5
            return self.F[:, t].toarray()[:,0]

        # manually make force matrix
        force_matrix = np.zeros(self.F.shape)
        force_matrix[:, 0] = self.F.toarray()[:,0]
        for i in range(1, force_matrix.shape[1]):
            force_matrix[:, i] = force_matrix[:, i-1]/2

        # calculate using custom load function
        res_func = NewmarkSolver()
        res_func.load_func = load_function
        res_func.beta = self.settings["beta"]
        res_func.gamma = self.settings["gamma"]
        res_func.initialise(self.number_eq, self.time)
        res_func.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # calculate using initial load matrix
        res_manual = NewmarkSolver()
        res_manual.beta = self.settings["beta"]
        res_manual.gamma = self.settings["gamma"]
        res_manual.initialise(self.number_eq, self.time)
        res_manual.calculate(self.M, self.C, self.K, force_matrix, 0, self.n_steps)

        # check if solutions are equal
        np.testing.assert_array_almost_equal(res_func.u, res_manual.u)



