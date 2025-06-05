import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import inv as sp_inv
from scipy.sparse import diags
from tqdm import tqdm

from solvers.base_solver import Solver
from solvers.utils import LumpingMethod


class CentralDifferenceSolver(Solver):
    """
    Central Difference Solver class. This class contains the explicit solver according to :cite:p: `Bathe_1996`.
    This class bases from :class:`~rose.model.solver.Solver`.

    :Attributes:
        - :self.is_lumped: bool which is true if mass matrix should be lumped
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

    def _create_diagonal_matrix(self, diag_elements, sparse=False):
        """
        Create diagonal matrix

        :param diag_elements: diagonal elements
        :param sparse: if True, return sparse matrix
        :return: diagonal matrix
        """
        if sparse:
            return diags(diag_elements, format='csc')
        return np.diagflat(diag_elements)

    def update_force(self, u, F_previous, t):
        """
        Updates the external force vector at time t

        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index
        :return:
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
        Perform calculation with the explicit central difference solver.

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        """

        self.initialise_stage(F)
        self.update_output_arrays(t_start_idx, t_end_idx)
        self.validate_input(t_start_idx, t_end_idx)

        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (t_end_idx - t_start_idx)

        self.update_rhs_at_time_step(t_start_idx)
        self.update_rhs_at_non_linear_iteration(t_start_idx, u=self.u0)

        M, C, K = self.check_for_sparse(M, C, K)

        if self.is_lumped:
            # Lump the mass matrix
            M_diag = self.lump_method.apply(M)
            C_diag = self.lump_method.apply(C)

            # Create diagonal matrices
            M = self._create_diagonal_matrix(M_diag, sparse=self._is_sparse_calculation)
            C = self._create_diagonal_matrix(C_diag, sparse=self._is_sparse_calculation)

            # Compute effective mass matrix
            M_till_diag = M_diag / (t_step ** 2) + C_diag / (2 * t_step)
            inv_M_till = 1.0 / M_till_diag

            # Compute constant matrices
            K_part = K - (2.0 / t_step ** 2) * M
            M_part = M_diag / (t_step ** 2) - C_diag / (2 * t_step)
        else:
            # Consistent mass formulation
            M_till = 1. / t_step ** 2 * M + 1 / (2 * t_step) * C
            if self._is_sparse_calculation:
                inv_M = sp_inv(M).tocsc()
                inv_M_till = sp_inv(M_till).tocsc()
            else:
                inv_M = inv(M)
                inv_M_till = inv(M_till)
            K_part = K - (2 / t_step ** 2) * M
            M_part = 1 / t_step ** 2 * M - 1 / (2 * t_step) * C

        # Initial conditions
        u = self.u0
        v = self.v0

        # Calculate initial acceleration
        if self.is_lumped:
            a = (self.F - K.dot(u) - C_diag * v) / M_diag
        else:
            a = inv_M.dot(self.F - K.dot(u) - C.dot(v))

        u_prev = u - t_step * v + 0.5 * t_step ** 2 * a

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a

        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")
        force = np.copy(self.F)

        for t in range(t_start_idx + 1, t_end_idx + 1):
            pbar.update(1)
            self.update_rhs_at_time_step(t, u=u)
            _, force = self.update_force(u, force, t)

            if self.is_lumped:
                internal_force_part_1 = K_part.dot(u)
                internal_force_part_2 = M_part * u_prev
                u_new = (force - internal_force_part_1 - internal_force_part_2) * inv_M_till
            else:
                internal_force_part_1 = K_part.dot(u)
                internal_force_part_2 = M_part.dot(u_prev)
                u_new = inv_M_till.dot(force - internal_force_part_1 - internal_force_part_2)

            # Calculate velocity and acceleration
            v = (u_new - u_prev) / (2 * t_step)
            a = (u_prev - 2 * u + u_new) / (t_step ** 2)

            if t == self.output_time_indices[t2]:
                self.u[t2, :] = u_new
                self.v[t2, :] = v
                self.a[t2, :] = a
                self.F_out[t2, :] = np.copy(self.F)
                t2 += 1

            u_prev = u
            u = u_new

        pbar.close()