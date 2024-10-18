
from solvers.base_solver import Solver

import numpy as np
from numpy.linalg import solve, inv
from scipy.sparse.linalg import splu, spsolve
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt

class NewmarkOgniBene(Solver):
    """
    Newmark Solver class. This class contains the implicit incremental Newmark solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    :Attributes:

       - :self.beta:     Newmark numerical stability parameter
       - :self.gamma:    Newmark numerical stability parameter
    """

    def __init__(self, Fy0, alpha_y, kb, cb, S0, Hd0, alpha_d, Hs0, alpha_s):
        super(NewmarkOgniBene, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5
        self.max_iter = 15
        self.tolerance = 1e-5

        self.Fb_max = 0.0
        # self.Fy = 70e3
        # self.Fy0 = 70e3
        self.Fy = Fy0
        self.Fy0 = Fy0
        self.alpha_y = alpha_y
        # self.alpha_y =12

        self.cb = cb
        self.kb = kb
        self.S = 0
        self.S0 = S0

        self.Hd = Hd0
        self.Hd0 = Hd0
        self.alpha_d = alpha_d

        self.Hs = Hs0
        self.Hs0 = Hs0
        self.alpha_s = alpha_s

        # self.cb = 0.12e6
        # self.kb = 210e6
        # self.S = 0
        # self.S0 = 0.004
        #
        # self.Hd = 0.6e9
        # self.Hd0 = 0.6e9
        # self.alpha_d =2.8
        #
        # self.Hs = 0.7e12
        # self.Hs0 = 0.7e12
        # self.alpha_s = 0.4

        self.H = 1/((1/self.Hd0)+(1/self.Hs0))

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

        # a = np.zeros(len(u))

        # initial acceleration
        if self._is_sparse_calculation:
            a = self.sparse_solver(m_global.astype(float), force_ini - c_part - k_part)
        else:
            a = inv(m_global).dot(force_ini - c_part - k_part)

        return a

    def update_force(self, u, v, F_previous, t):
        """
        Updates the external force vector at time t

        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index
        :return:
        """
        ballast_indices = [1,2]

        u_ballast = u[ballast_indices[0]] -  u[ballast_indices[1]]
        u_ballast_prev = self.u_prev[ballast_indices[0]] -  self.u_prev[ballast_indices[1]]
        v_ballast = v[ballast_indices[0]] -  v[ballast_indices[1]]

        # F_copy = np.copy(self.F)

        alpha = self.cb/(self.cb + self.t_step*(self.kb+self.H))
        beta = ( 1+ self.t_step*self.kb/self.cb)**-1

        # self.F = self.cb/(beta * self.t_step) *du - self.cb/self.t_step
        # self.F_ballast = self.kb * u[ballast_index] + self.cb * v[ballast_index]

        self.F_ballast = self.cb/(beta * self.t_step) *u_ballast - self.cb/self.t_step *u_ballast_prev

        du_star = u_ballast
        # if self.F_ballast > self.Fy and self.F_ballast > self.F_ballast_prev:
        if self.F_ballast > self.Fy and du_star > 0:
            # du_star = du - self.S

            self.F_ballast = alpha * self.F_ballast_prev + alpha* self.t_step *(self.H * v_ballast + self.kb/self.cb * self.H *du_star + self.kb/self.cb*self.Fy)
            self.delta_up = self.delta_up_prev + (self.F_ballast - self.F_ballast_prev) / self.H

            # new_k = self.F_ballast / u_ballast

            # new_c = self.F_ballast / v[ballast_index]

            # # update only the ballast stiffness matrix
            # self.K[ballast_indices[0], ballast_indices[0]] = self.K[ballast_indices[0], ballast_indices[0]] - self.kb + new_k
            # self.K[ballast_indices[1], ballast_indices[1]] = self.K[ballast_indices[1], ballast_indices[1]] - self.kb + new_k
            # self.K[ballast_indices[0], ballast_indices[1]] = self.K[ballast_indices[0], ballast_indices[1]] + self.kb - new_k
            # self.K[ballast_indices[1], ballast_indices[0]] = self.K[ballast_indices[1], ballast_indices[0]] + self.kb - new_k
            #
            # self.kb = new_k
            # tmp = 1+1


        else:
            self.delta_up = self.delta_up_prev



        # calculates force with custom load function
        # self.update_rhs_at_non_linear_iteration(t,u=u)

        force = self.F

        self.Fb_max = np.max([self.F_ballast, self.Fb_max])

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)


        return d_force, F_total, self.K

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx, n_cycles):

        all_S = []
        all_cycles = range(n_cycles)
        self.S = self.S0
        for i in range(n_cycles):
            self.Fb_max = self.Fy0
            # self.Fb_max = 157e3
            self.calculate_cycle(M, C, K, F, t_start_idx, t_end_idx)
            self.S = self.S + self.delta_up
            all_S.append(self.S)

            # rate = np.log10(self.S/self.S0)
            rate = np.log(self.S/self.S0)
            time_var = self.S-self.S0

            self.Hd = self.Hd0 *np.exp(self.alpha_d * rate *time_var)
            self.Hs = self.Hs0 *np.exp(self.alpha_s * rate *time_var)

            dhd = self.Hd - self.Hd0
            dhs = self.Hs - self.Hs0

            self.H = 1 / ((1 / self.Hd) + (1 / self.Hs))

            self.Fy = self.Fy0  + (self.Fb_max - self.Fy0) * (1-(1/(1+self.alpha_y*rate)))

            K = np.copy(self.K)

        # plt.plot(all_cycles,all_S)
        plt.semilogx(all_cycles, all_S)
        plt.xlim(10,10000)
        plt.ylim(0.025, 0)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.grid(True, which="major", ls="-", alpha=0.5)
        # plt.xlabel('X (log scale)')
        # plt.ylabel('Y')
        # plt.title('Plot with Logarithmic X-Axis')
        # plt.legend()

        # Add minor grid lines
        plt.minorticks_on()

        plt.show()

        a=1+1


    def calculate_cycle(self, M, C, K, F, t_start_idx, t_end_idx):
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

        self.initialise_stage(np.copy(F))

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        self.M = M
        self.C = C
        self.K = K

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

        self.u_prev = np.copy(u)

        self.delta_up = 0.0
        self.delta_up_prev = 0.0
        self.F_ballast_prev = 0.0

        self.update_rhs_at_time_step(t_start_idx)
        self.update_rhs_at_non_linear_iteration(t_start_idx,u=u)

        # initial force conditions: for computation of initial acceleration

        d_force = self.F

        # a = self.calculate_initial_acceleration(M, C, K, d_force, u, v)
        a=np.zeros_like(u)

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
                d_force, F_previous_i, new_K = self.update_force(u,v, F_previous, t)

                # update stiffness matrix
                # K_till = new_K + C * (gamma / (beta * self.t_step)) + M * (1 / (beta * self.t_step ** 2))

                # if self._is_sparse_calculation:
                #     inv_K_till = splu(K_till)
                # else:
                #     inv_K_till = inv(K_till)

                # external force
                force_ext = d_force + m_part + c_part

                # solve
                # if self._is_sparse_calculation:
                #     du = inv_K_till.solve(force_ext - force_previous)
                # else:
                #     du = inv_K_till.dot(force_ext - force_previous)

                if self._is_sparse_calculation:
                    du = spsolve(K_till, force_ext - force_previous)
                else:
                    du = solve(K_till, force_ext - force_previous)
                    # du = inv_K_till.dot(force_ext - force_previous)

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
            self.delta_up_prev = np.copy(self.delta_up)
            self.F_ballast_prev = np.copy(self.F_ballast)
            self.u_prev = np.copy(u)

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

