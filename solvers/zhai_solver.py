from solvers.base_solver import Solver

import os
import pickle

import numpy as np
from numpy.linalg import solve, inv
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv as sp_inv
from scipy.sparse import issparse, csc_matrix

from tqdm import tqdm
import logging


class ZhaiSolver(Solver):
    """
    Zhai Solver class. This class contains the explicit solver according to [Zhai 1996]. This class bases from
    :class:`~rose.model.solver.Solver`.

    :Attributes:

       - :self.psi:      Zhai numerical stability parameter
       - :self.phi:      Zhai numerical stability parameter
       - :self.beta:     Newmark numerical stability parameter
       - :self.gamma:    Newmark numerical stability parameter
    """
    def __init__(self):
        super(ZhaiSolver, self).__init__()

        self.psi = 0.5
        self.phi = 0.5
        self.beta = 1/4
        self.gamma = 1/2

    def calculate_initial_values(self, M, C, K, F, u0, v0):
        """
        Calculate inverse mass matrix and initial acceleration

        :param M: global mass matrix
        :param C: global damping matrix
        :param K: global stiffness matrix
        :param F: global force vector at current time step
        :param u0: initial displacement
        :param v0:  initial velocity
        :return:
        """
        if self._is_sparse_calculation:
            inv_M = sp_inv(M).tocsr()
        else:
            inv_M = inv(M)

        a0 = self.evaluate_acceleration(inv_M, C, K, F, u0, v0)
        return inv_M, a0

    def calculate_force(self, u, t):
        """
        Calculate external force if a load function is given. If no load function is given, force is taken from current
        load vector

        :param u: displacement at time t
        :param F: External force matrix
        :param t: current time step
        :return:
        """

        # calculates force with custom load function
        self.update_non_linear_iteration_rhs(t, u=u)

        force = self.F
        # Convert force vector to a 1d numpy array
        if issparse(force):
            force = force.toarray()[:, 0]

        return force

    def prediction(self, u, v, a, a_old, dt, is_initial):
        """
        Perform prediction for displacement and acceleration

        :param u: displacement
        :param v: velocity
        :param a: acceleration
        :param a_old: acceleration at previous time step
        :param dt: time step size
        :param is_initial: bool to indicate current iteration is the initial iteration
        :return:
        """

        # set Zhai factors
        if is_initial:
            psi = phi = 0
        else:
            psi = self.psi
            phi = self.phi

        # predict displacement and velocity
        u_new = u + v * dt + (1/2 + psi) * a * dt ** 2 - psi * a_old * dt**2
        v_new = v + (1 + phi) * a * dt - phi * a_old * dt
        return u_new, v_new

    @staticmethod
    def evaluate_acceleration(inv_M, C, K, F, u, v):
        """
        Calculate acceleration

        :param inv_M: inverse global mass matrix
        :param C: global damping matrix
        :param K: Global stiffness matrix
        :param F: Force vector at current time step
        :param u: displacement
        :param v: velocity
        :return:
        """
        a_new = inv_M.dot(F - K.dot(u) - C.dot(v))
        return a_new

    def newmark_iteration(self, u, v, a, a_new, dt):
        """
        Perform Newmark iteration as corrector for displacement and velocity

        :param u: displacement
        :param v: velocity
        :param a: acceleration
        :param a_new: predicted acceleration
        :param dt: delta time
        :return:
        """
        u_new = u + v * dt + (1/2 - self.beta) * a * dt ** 2 + self.beta * a_new * dt ** 2
        v_new = v + (1-self.gamma) * a * dt + self.gamma * a_new * dt

        return u_new, v_new

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Perform calculation with the explicit Zhai solver [Zhai 1996]

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        :return:
        """

        self.initialise_stage(F)

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        # validate input
        self.validate_input(t_start_idx, t_end_idx)

        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # initial force conditions: for computation of initial acceleration
        self.update_time_step_rhs(t_start_idx)
        self.update_non_linear_iteration_rhs(t_start_idx, u=self.u0)

        if issparse(self.F):
            force = self.F.toarray()[:, 0]
        else:
            force = self.F

        # get initial displacement, velocity, acceleration and inverse mass matrix
        u = self.u0
        v = self.v0
        inv_M, a = self.calculate_initial_values(M, C, K, force, u, v)

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a

        a_old = np.zeros(self.number_equations)

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        is_initial = True
        for t in range(t_start_idx + 1, t_end_idx + 1):
            self.update_time_step_rhs(t, u=u)

            # update progress bar
            pbar.update(1)

            # check if current timestep is the initial timestep
            if t > 1:
                is_initial = False

            # Predict displacement and velocity
            u_new, v_new = self.prediction(u, v, a, a_old, t_step, is_initial)

            # Calculate predicted external force vector
            force = self.calculate_force(u_new, t)

            # Calculate predicted acceleration
            a_new = self.evaluate_acceleration(inv_M, C, K, force, u_new, v_new)

            # Correct displacement and velocity
            u_new, v_new = self.newmark_iteration(u, v, a, a_new, t_step)

            # Calculate corrected force vector
            force = self.calculate_force(u_new, t)

            # Calculate corrected acceleration
            a_new = self.evaluate_acceleration(inv_M, C, K, force, u_new, v_new)

            # add to results
            if t == self.output_time_indices[t2]:
                self.u[t2, :] = u_new
                self.v[t2, :] = v_new
                self.a[t2, :] = a_new

                t2 += 1

            # set vectors for next time step
            u = np.copy(u_new)
            v = np.copy(v_new)

            a_old = np.copy(a)
            a = np.copy(a_new)

        # close the progress bar
        pbar.close()
