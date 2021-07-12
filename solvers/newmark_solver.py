from solvers.base_solver import Solver

import numpy as np
from numpy.linalg import solve, inv
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv as sp_inv
from scipy.sparse import issparse, csc_matrix
import os
import pickle
from tqdm import tqdm
import logging


class NewmarkSolver(Solver):
    """
    Newmark Solver class. This class contains the implicit incremental Newmark solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    :Attributes:

       - :self.beta:     Newmark numerical stability parameter
       - :self.gamma:    Newmark numerical stability parameter
    """

    def __init__(self):
        super(NewmarkSolver, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5

    def calculate_initial_acceleration(self, m_global, c_global, k_global, force_ini, u, v):
        r"""
        Calculation of the initial conditions - acceleration for the first time-step.

        :param m_global: Global mass matrix
        :param c_global: Global damping matrix
        :param k_global: Global stiffness matrix
        :param force_ini: Initial force
        :param u: Initial conditions - displacement
        :param v: Initial conditions - velocity

        :return a: Initial acceleration
        """

        k_part = k_global.dot(u)
        c_part = c_global.dot(v)

        # initial acceleration
        if self._is_sparse_calculation:
            a = sp_inv(m_global).dot(force_ini - c_part - k_part)
        else:
            a = inv(m_global).dot(force_ini - c_part - k_part)

        return a

    def update_force(self, u, F_previous, t):
        """
        Updates the external force vector at time t

        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index
        :return:
        """

        # calculates force with custom load function
        force = self.load_func(u, t)

        # Convert force vector to a 1d numpy array
        if issparse(force):
            force = force.toarray()[:, 0]

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)

        return d_force, F_total

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Newmark integration scheme.
        Incremental formulation.

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
        """

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        # validate solver index
        self.validate_input(F, t_start_idx, t_end_idx)

        # calculate time step size
        # todo correct t_step, as it is not correct, but tests succeed
        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial force conditions: for computation of initial acceleration
        if issparse(F):
            d_force = F[:, t_start_idx].toarray()[:, 0]
        else:
            d_force = F[:, t_start_idx]

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        a = self.calculate_initial_acceleration(M, C, K, d_force, u, v)

        # initialise delta velocity
        dv = np.zeros(len(v))

        # add to results initial conditions
        self.u[t_start_idx, :] = u
        self.v[t_start_idx, :] = v
        self.a[t_start_idx, :] = a
        self.f[t_start_idx, :] = d_force

        # combined stiffness matrix
        K_till = K + C.dot(gamma / (beta * t_step)) + M.dot(1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        # initialise Force from load function
        if self.load_func is not None and issparse(F):
            F_previous = F[:, t_start_idx].toarray()[:, 0]
        else:
            F_previous = F[:, t_start_idx]

        # initialise absorbing boundary if not initialised
        if self.absorbing_boundary is None:
            if self._is_sparse_calculation:
                self.absorbing_boundary = csc_matrix(K.shape)
            else:
                self.absorbing_boundary = np.zeros(K.shape)

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = v.dot(1 / (beta * t_step)) + a.dot(1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v.dot(gamma / beta) + a.dot(t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # update external force
            if self.load_func is not None:
                d_force, F_previous = self.update_force(u, F_previous, t)
            else:
                d_force = F[:, t] - F_previous
                if issparse(d_force):
                    d_force = d_force.toarray()[:, 0]
                F_previous = F[:, t]

            # external force
            force_ext = d_force + m_part + c_part - self.absorbing_boundary * dv

            # solve
            if self._is_sparse_calculation:
                du = spsolve(K_till, force_ext)
            else:
                du = solve(K_till, force_ext)

                # velocity calculated through Newmark relation
            dv = (
                du.dot(gamma / (beta * t_step))
                - v.dot(gamma / beta)
                + a.dot(t_step * (1 - gamma / (2 * beta)))
            )

            # acceleration calculated through Newmark relation
            da = (
                du.dot(1 / (beta * t_step ** 2))
                - v.dot(1 / (beta * t_step))
                - a.dot(1 / (2 * beta))
            )

            # update variables
            u = u + du
            v = v + dv
            a = a + da

            # add to results
            self.u[t, :] = u
            self.v[t, :] = v
            self.a[t, :] = a

        # calculate nodal force
        self.f[:, :] = np.transpose(K.dot(np.transpose(self.u)))
        # close the progress bar
        pbar.close()

