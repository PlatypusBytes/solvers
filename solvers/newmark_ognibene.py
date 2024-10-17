
from solvers.base_solver import Solver

import numpy as np
from numpy.linalg import solve, inv
from scipy.sparse.linalg import splu
from tqdm import tqdm
import logging

class NewmarkOgniBene(Solver):
    """
    Newmark Solver class. This class contains the implicit incremental Newmark solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    :Attributes:

       - :self.beta:     Newmark numerical stability parameter
       - :self.gamma:    Newmark numerical stability parameter
    """

    def __init__(self):
        super(NewmarkOgniBene, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5
        self.max_iter = 15
        self.tolerance = 1e-5

        self.Fy = 0.07
        self.alpha_y =12

        self.cb = 0.12
        self.kb = 210
        self.S = 0
        self.S0 = 0

        self.Hd = 0.6
        self.alpha_d =2.8

        self.Hs = 0.7e3
        self.alpha_s = 0.4

        self.H = 1/((1/self.Hd)+(1/self.Hs))

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

        a = np.zeros(len(u))

        # # initial acceleration
        # if self._is_sparse_calculation:
        #     a = self.sparse_solver(m_global.astype(float), force_ini - c_part - k_part)
        # else:
        #     a = inv(m_global).dot(force_ini - c_part - k_part)

        return a

    def update_force(self, du, dv, F_previous, t):
        """
        Updates the external force vector at time t

        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index
        :return:
        """

        F_copy = np.copy(self.F)

        alpha = self.cb/(self.cb + self.t_step*(self.kb+self.H))
        beta = ( 1+ self.t_step*self.kb/self.cb)**-1

        self.F = self.cb/(beta * self.t_step) *du - self.cb/self.t_step


        self.F = alpha * F_copy + alpha* self.t_step *(self.H * dv + self.kb/self.cb * self.H *(du - self.S) + self.kb/self.cb*self.Fy)

        self.delta_up = self.delta_up + (self.F - F_previous) /self.H

        # calculates force with custom load function
        # self.update_rhs_at_non_linear_iteration(t,u=u)

        force = self.F

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)

        accum_disp = np.zeros(len(du))
        K = np.zeros((len(du), len(du)))
        C = np.zeros((len(du), len(du)))





        return d_force, F_total, accum_disp, K, C

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Newmark implicit integration scheme with Newton Raphson strategy for non-linear force.
        Incremental formulation.

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
        """

        self.initialise_stage(F)

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        self.update_output_arrays(t_start_idx, t_end_idx)
        # validate solver index
        self.validate_input(t_start_idx, t_end_idx)

        # calculate time step size
        self.t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial conditions u, v, a
        u = self.u0
        v = self.v0

        self.delta_up = np.zeros(len(u))

        self.update_rhs_at_time_step(t_start_idx)
        self.update_rhs_at_non_linear_iteration(t_start_idx,u=u)

        # initial force conditions: for computation of initial acceleration

        d_force = self.F

        a = self.calculate_initial_acceleration(M, C, K, d_force, u, v)
        a=0

        # initialise delta velocity
        dv = np.zeros(len(v))

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a
        self.f[output_time_idx, :] = d_force
        self.F_out[output_time_idx, :] = np.copy(self.F)

        # combined stiffness matrix
        K_till = K + C * (gamma / (beta * self.t_step)) + M * (1 / (beta * self.t_step ** 2))

        if self._is_sparse_calculation:
            inv_K_till = splu(K_till)
        else:
            inv_K_till = inv(K_till)

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        # initialise Force from load function
        F_previous = np.copy(self.F)

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):
            self.update_rhs_at_time_step(t, u=u)

            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = v * (1 / (beta * self.t_step)) + a * (1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v * (gamma / beta) + a * (self.t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # set ext force from previous time iteration
            force_ext_prev = d_force + m_part + c_part

            # initialise
            du_tot = np.zeros(len(u))
            i = 0
            force_previous = 0

            # Newton Raphson loop where force is updated in every iteration
            converged = False
            while not converged and i < self.max_iter:

                # update external force
                d_force, F_previous_i, accum_disp, new_K, new_C = self.update_force(du_tot,dv, F_previous, t)
                # external force
                force_ext = d_force + m_part + c_part

                # solve
                if self._is_sparse_calculation:
                    du = inv_K_till.solve(force_ext - force_previous)
                else:
                    du = inv_K_till.dot(force_ext - force_previous)

                # set du for first iteration
                if i == 0:
                    du_ini = np.copy(du)

                # energy converge criterion according to bath 1996, chapter 8.4.4
                error = np.linalg.norm(du * force_ext) / np.linalg.norm(du_ini * force_ext_prev)
                converged = (error < self.tolerance)

                # calculate total du for current time step
                du_tot += du

                # velocity calculated through Newmark relation
                dv = (
                        du_tot * (gamma / (beta * self.t_step))
                        - v * (gamma / beta)
                        + a * (self.t_step * (1 - gamma / (2 * beta)))
                )

                u = u + du

                if not converged:
                    force_previous = np.copy(force_ext)

                i += 1

            # acceleration calculated through Newmark relation
            da = (
                    du_tot * (1 / (beta * self.t_step ** 2))
                    - v * (1 / (beta * self.t_step))
                    - a * (1 / (2 * beta))
            )

            F_previous = F_previous_i
            # update variables

            v = v + dv
            a = a + da

            # add to results
            if t == self.output_time_indices[t2]:
                self.u[t2, :] = u
                self.v[t2, :] = v
                self.a[t2, :] = a

                self.F_out[t2, :] = np.copy(self.F)
                t2 += 1

        # calculate nodal force
        self.f[:, :] = np.transpose(K.dot(np.transpose(self.u)))
        # close the progress bar
        pbar.close()

