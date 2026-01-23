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




    def test_load_function_explicit(self):
        """
        Test if Newmark solver returns equal results while using a load function and an initial force matrix. The load
        function is chosen such that at each time step it calculates the same value as the tested force matrix at that
        time step.
        :return:
        """

        def load_function(t,u=None):
            # half load each time step
            if t>0:
                self.F[:, t] = self.F[:,t-1] * 0.5
            return self.F[:, t].toarray()[:,0]

        # manually make force matrix
        force_matrix = np.zeros(self.F.shape)
        force_matrix[:, 0] = self.F.toarray()[:,0]
        for i in range(1, force_matrix.shape[1]):
            force_matrix[:, i] = force_matrix[:, i-1]/2

        # calculate using custom load function
        res_func = NewmarkExplicit()
        res_func.update_rhs_at_time_step_func = load_function
        res_func.beta = self.settings["beta"]
        res_func.gamma = self.settings["gamma"]
        res_func.initialise(self.number_eq, self.time)
        res_func.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # calculate using initial load matrix
        res_manual = NewmarkExplicit()
        res_manual.beta = self.settings["beta"]
        res_manual.gamma = self.settings["gamma"]
        res_manual.initialise(self.number_eq, self.time)
        res_manual.calculate(self.M, self.C, self.K, force_matrix, 0, self.n_steps)

        # check if solutions are equal
        np.testing.assert_array_almost_equal(res_func.u, res_manual.u)

    @pytest.mark.workinprogress
    def test_load_function_implicit(self):
        """
        Test if Newmark solver returns equal results while using a load function and implicit Newmark solver and an
        initial force matrix. The load function is chosen such that at each time step it calculates the same value as
        the tested force matrix at that time step.
        :return:
        """

        def load_function(u,t):
            # force is a function of the displacement
            F = np.sin(u) * 10

            return F

        # calculate using custom load function and implicit Newmark solver
        res_func = NewmarkImplicitForce()
        res_func.load_func = load_function
        res_func.beta = self.settings["beta"]
        res_func.gamma = self.settings["gamma"]
        res_func.initialise(2, self.time)
        res_func.u0 = [np.pi/2,np.pi/2]
        res_func.u0 = [0, 0]

        F = np.ones((2,len(self.time)))*10
        res_func.calculate(self.M, self.C, self.K, F, 0, self.n_steps)

    def test_output_interval_newmark_explicit(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)

        # reshape force vector
        F = np.zeros((2, self.n_steps + 1))
        F[1, :] = 10
        self.F = sparse.csc_matrix(np.array(F))

        output_interval = 10

        # write all output
        res = NewmarkExplicit()
        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        expected_displacement = np.concatenate((res.u[0::output_interval, :], res.u[None, -1, :]), axis=0)

        # write every other step
        res_2 = NewmarkExplicit()
        res_2.output_interval = output_interval
        res_2.initialise(self.number_eq, self.time)
        res_2.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, res_2.u)

    def test_output_interval_newmark_implicit(self):
        self.M, self.K, self.C, self.F = set_matrices_as_np_array(self.M, self.K, self.C, self.F)

        # reshape force vector
        F = np.zeros((2, self.n_steps + 1))
        F[1, :] = 10
        self.F = sparse.csc_matrix(np.array(F))

        output_interval = 10

        # write all output
        res = NewmarkImplicitForce()
        res.initialise(self.number_eq, self.time)
        res.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)
        expected_displacement = np.concatenate((res.u[0::output_interval, :], res.u[None, -1, :]), axis=0)

        # write every other step
        res_2 = NewmarkImplicitForce()
        res_2.output_interval = output_interval
        res_2.initialise(self.number_eq, self.time)
        res_2.calculate(self.M, self.C, self.K, self.F, 0, self.n_steps)

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, res_2.u)

