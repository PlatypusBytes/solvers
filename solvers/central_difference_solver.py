import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import inv as sp_inv
from scipy.sparse import issparse
from tqdm import tqdm

from solvers.base_solver import Solver
from solvers.utils import LumpingMethod


class CentralDifferenceSolver(Solver):
    """
    Central Difference Solver class. This class contains the explicit solver according to :cite:p: `Bathe_1996`.
    This class bases from :class:`~rose.model.solver.Solver`.

    :Attributes:
        - :self.is_lumped: bool which is true if mass matrix should be lumped and damping matrix is to be neglected
        - :self.lump_method: method of lumping the mass matrix
    """

    def __init__(self, lumped=True, lumping_method=LumpingMethod.RowSum):
        """
        Initialisation of the Central Difference Solver class.

        :param lumped: mass matrix lumped: default True
        :param lumping_method: method of lumping the mass matrix: default "RowSum"
        """

        super(CentralDifferenceSolver, self).__init__()

        self.is_lumped = lumped
        if not isinstance(lumping_method, LumpingMethod):
            raise ValueError("Lumping method must be of type LumpingMethod")
        self.lump_method = lumping_method

    def calculate_force(self, u, t):
        """
        Calculate external force if a load function is given. If no load function is given, force is taken from current
        load vector

        :param u: displacement at time t
        :param t: current time step
        :return: external force vector
        """

        # calculates force with custom load function
        self.update_rhs_at_non_linear_iteration(t, u=u)

        force = self.F
        # Convert force vector to a 1d numpy array
        if issparse(force):
            force = force.toarray()[:, 0]

        return force

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Perform calculation with the explicit central difference solver.

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        :return:
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
            M = self.lump_method.apply(M)
            C = self.lump_method.apply(C)
            inv_M = 1 / M
            M = np.diagflat(M)
            C = np.diagflat(C)
            inv_M = np.diagflat(inv_M)
        else:
            if self._is_sparse_calculation:
                inv_M = sp_inv(M).tocsc()
            else:
                inv_M = inv(M)

        # get initial displacement, velocity, acceleration
        u = self.u0
        v = self.v0
        a = inv_M.dot(self.F - K.dot(u) - C.dot(v))
        u_prev = u - t_step * v + 1 / 2 * t_step ** 2 * a

        # Effective mass matrix
        M_till = 1 / t_step ** 2 * M + 1 / (2 * t_step) * C
        if self.is_lumped:
            inv_M_till = np.diagflat(1 / np.diag(M_till))
        else:
            if self._is_sparse_calculation:
                inv_M_till = sp_inv(M_till).tocsc()
            else:
                inv_M_till = inv(M_till)

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            # Calculate predicted external force vector
            force = self.calculate_force(u, t)

            # calculate displacement at new time step
            if self.is_lumped:
                internal_force_part_1 = np.squeeze(np.asarray((K - (2 / t_step**2) * M).dot(u)))
            else:
                internal_force_part_1 = (K - 2 / t_step**2 * M).dot(u)
            internal_force_part_2 = (1 / t_step**2 * M - 1 / (2 * t_step) * C).dot(u_prev)
            u_new = inv_M_till.dot(force - internal_force_part_1 - internal_force_part_2)

            # calculate velocity and acceleration at current time step
            v = 1 / (2 * t_step) * (-u_prev + u_new)
            a = 1 / t_step**2 * (u_prev - 2 * u + u_new)

            # add to results
            if t == self.output_time_indices[t2]:
                # a and v are calculated at previous time step
                self.u[t2, :] = u_new
                self.v[t2, :] = v
                self.a[t2, :] = a

                self.F_out[t2, :] = np.copy(self.F)

                t2 += 1

            # set vectors for next time step
            u_prev = np.copy(u)
            u = np.copy(u_new)

        # close the progress bar
        pbar.close()
