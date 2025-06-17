import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import inv as sp_inv
from tqdm import tqdm

from solvers.base_solver import Solver
from solvers.utils import LumpingMethod


class BatheSolver(Solver):
    """
    Bathe Solver class. This class contains the explicit solver according to :cite:p: `Noh_Bathe_2013`.

    :Attributes:
        - :self.is_lumped: bool which is true if mass matrix should be lumped
        - :self.lump_method: method of lumping the mass matrix
        - :self._p: Bathe parameter for the integration
    """

    def __init__(self, lumping_method=LumpingMethod.RowSum):
        """
        Initialisation of the Bathe Solver class.

        Parameters:
        :param lumping_method: method of lumping the mass matrix: default "RowSum"
        """

        super(BatheSolver, self).__init__()

        if not isinstance(lumping_method, LumpingMethod):
            raise ValueError("Lumping method must be of type LumpingMethod")
        self.lump_method = lumping_method
        self.is_lumped = lumping_method != LumpingMethod.NONE
        self._p = 0.54  # Bathe parameter

    def update_force(self, u, F_previous, t):
        """
        Updates the external force vector at time t

        Parameters:
        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index
        :return: incremental force vector and total force vector
        """

        # calculates force with custom load function
        self.update_rhs_at_non_linear_iteration(t,u=u)

        force = self.F

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)

        return d_force, F_total

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Perform calculation with the Bathe solver.

        Parameters:
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        """

        self.initialise_stage(F)
        self.update_output_arrays(t_start_idx, t_end_idx)
        # validate input
        self.validate_input(t_start_idx, t_end_idx)

        # calculate step size
        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (t_end_idx - t_start_idx)

        # initial force conditions: for computation of initial acceleration
        self.update_rhs_at_time_step(t_start_idx)
        self.update_rhs_at_non_linear_iteration(t_start_idx, u=self.u0)

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        if self.is_lumped:
            # Lumping only applies to the mass matrix the damping matrix remain consistent
            M_diag = self.lump_method.apply(M)
            inv_M_diag = 1 / M_diag
        else:
            # inverse mass matrix
            if self._is_sparse_calculation:
                inv_M = sp_inv(M).tocsc()
            else:
                inv_M = inv(M)

        # compute constants
        q1 = (1. - 2 * self._p) / (2 * self._p * (1 - self._p))
        q2 = 1 / 2 - self._p * q1
        q0 = -q1 - q2 + 1 / 2
        a0 = self._p * t_step
        a1 = 1 / 2 * (self._p * t_step) ** 2
        a2 = a0 / 2
        a3 = (1 - self._p) * t_step
        a4 = 1 / 2 * ((1 - self._p) * t_step) ** 2
        a5 = q0 * a3
        a6 = (1 / 2 + q1) * a3
        a7 = q2 * a3

        # get initial displacement, velocity, acceleration
        u = self.u0
        v = self.v0
        if self.is_lumped:
            a = inv_M_diag * (self.F - K.dot(u) - C.dot(v))
        else:
            a = inv_M.dot(self.F - K.dot(u) - C.dot(v))

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        # initialise Force from load function
        force = np.copy(self.F)
        force_previous = np.copy(self.F)

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            self.update_rhs_at_time_step(t, u=u)

            # update external force
            _, force = self.update_force(u, force, t)

            # first sub-step
            u_t_p = u + a0 * v + a1 * a
            force_term = (1 - self._p) * force_previous + self._p * force
            force_term = force_term - K.dot(u_t_p) - C.dot(v + a0 * a)

            if self.is_lumped:
                a_t_p = inv_M_diag * force_term
            else:
                a_t_p = inv_M.dot(force_term)
            v_t_p = v + a2 * (a + a_t_p)

            # second sub-step
            u = u_t_p + a3 * v_t_p + a4 * a_t_p
            force_term = force - K.dot(u) - C.dot(v_t_p + a3 * a_t_p)
            if self.is_lumped:
                a_next = inv_M_diag * force_term
            else:
                a_next = inv_M.dot(force_term)
            v = v_t_p + a5 * a + a6 * a_t_p + a7 * a_next
            a = a_next
            force_previous = np.copy(force)

            # add to results
            if t == self.output_time_indices[t2]:
                # a and v are calculated at previous time step
                self.u[t2, :] = u
                self.v[t2, :] = v
                self.a[t2, :] = a
                self.F_out[t2, :] = np.copy(self.F)
                t2 += 1

        # close the progress bar
        pbar.close()
