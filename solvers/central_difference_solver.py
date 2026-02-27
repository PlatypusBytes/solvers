import numpy as np
import numpy.typing as npt
from scipy.sparse import diags, issparse
from tqdm import tqdm
from typing import Optional

from solvers.utils import LumpingMethod
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType
from solvers.preconditioners import PreconditionerABC
from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolverLU

class CentralDifferenceSolver(BaseSolverABC):
    """
    Explicit central difference solver following the :cite:p:`Bathe_1996` formulation.
    """
    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None,
                 lumping_method: LumpingMethod = LumpingMethod.DiagonalScaling
                 ):
        """
        Constructor of the Central Difference Solver.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            linear_solver (Optional[LinearSolversABC]): Linear solver to be used (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner to be used (default: None)
            lumping_method (LumpingMethod): Lumping method to be used (default: DiagonalScaling)
        """

        if not isinstance(lumping_method, LumpingMethod):
            raise ValueError("Lumping method must be of type LumpingMethod")

        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()
        self.lump_method = lumping_method
        self.is_lumped = lumping_method != LumpingMethod.NONE
        self._is_sparse_calculation = False

    @property
    def type(self) -> TimeIntegrationType:
        """
        Time integration type of the solver.

        Returns:
            TimeIntegrationType: The time integration type of the solver.
        """
        return TimeIntegrationType.DYNAMIC

    def initialise(self, number_eq: int, time: npt.NDArray[np.float64]):
        """
        Initialise solver state for the provided number of equations and time vector.

        Args:
            number_eq (int): Number of equations
            time (npt.NDArray[np.float64]): Time vector
        """
        self.state.initialise(number_eq, time)

    def __create_diagonal_matrix(self, diag_elements: np.ndarray) -> Matrix:
        """
        Create a diagonal matrix with the provided diagonal entries.

        Args:
            diag_elements (np.ndarray): Diagonal elements of the matrix

        Returns:
            Matrix: Diagonal matrix with the provided elements
        """

        if self._is_sparse_calculation:
            return diags(diag_elements, format="csc")
        return np.diagflat(diag_elements)

    def calculate(self, M: Matrix, C: Matrix, K: Matrix, F: Matrix, t_start_idx: int, t_end_idx: int):
        """
        Perform calculation with the explicit central difference solver.

        Args:
            M (Matrix): Mass matrix
            C (Matrix): Damping matrix
            K (Matrix): Stiffness matrix
            F (Matrix): External force matrix
            t_start_idx (int): Start time index
            t_end_idx (int): End time index
        """

        # initialize force for the stage
        self.force.initialise_stage(F)
        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

        # update output arrays for the stage
        self.state.update_output_arrays(t_start_idx, t_end_idx)

        # calculate time step size
        t_step = (self.state.time[t_end_idx] - self.state.time[t_start_idx]) / (t_end_idx - t_start_idx)

        # initial force conditions: for computation of initial acceleration
        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=self.state.u0)

        M, C, K = self.state.check_for_sparse(M, C, K)
        self._is_sparse_calculation = any(issparse(mat) for mat in (M, C, K))

        if self.is_lumped:
            # Lump the mass matrix
            M_diag = self.lump_method.apply(M)
            C_diag = self.lump_method.apply(C)

            # Create diagonal matrices
            M = self.__create_diagonal_matrix(M_diag)
            C = self.__create_diagonal_matrix(C_diag)

            # Compute effective mass matrix
            M_till_diag = M_diag / (t_step ** 2) + C_diag / (2 * t_step)
            inv_M_till = 1.0 / M_till_diag

            # Compute constant matrices
            K_part = K - (2.0 / t_step ** 2) * M
            M_part = M_diag / (t_step ** 2) - C_diag / (2 * t_step)
        else:
            # Consistent mass formulation
            M_till = 1. / t_step ** 2 * M + 1 / (2 * t_step) * C
            K_part = K - (2 / t_step ** 2) * M
            M_part = 1 / t_step ** 2 * M - 1 / (2 * t_step) * C

        # Initial conditions
        u = self.state.u0
        v = self.state.v0

        # Build preconditioner if provided
        if self.preconditioner is not None and not self.is_lumped:
            pre_c = self.preconditioner.build(M)
        else:
            pre_c = None

        # Calculate initial acceleration
        if self.is_lumped:
            a = (self.force.F - K.dot(u) - C_diag * v) / M_diag
        else:
            a = calculate_initial_acceleration(M, C, K, self.force.F, u, v, self.linear_solver, self.preconditioner)

        u_prev = u - t_step * v + 0.5 * t_step ** 2 * a

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, v, a, K.dot(u), self.force.F)

        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")
        force_previous = np.copy(self.force.F)

        # Build preconditioner if provided
        if self.preconditioner is not None and not self.is_lumped:
            pre_c = self.preconditioner.build(M_till)
        else:
            pre_c = None

        for t in range(t_start_idx + 1, t_end_idx + 1):

            # update progress bar
            pbar.update(1)

            # update force at time step
            self.force.update_rhs_at_time_step(t, u=u)
            # update external force
            _, force_current = self.force.update_force(u, force_previous, t)

            if self.is_lumped:
                internal_force_part_1 = K_part.dot(u)
                internal_force_part_2 = M_part * u_prev
                u_new = (force_current - internal_force_part_1 - internal_force_part_2) * inv_M_till
            else:
                internal_force_part_1 = K_part.dot(u)
                internal_force_part_2 = M_part.dot(u_prev)
                u_new = self.linear_solver.solve(M_till, force_current - internal_force_part_1 - internal_force_part_2, M=pre_c)

            # Calculate velocity and acceleration
            v = (u_new - u_prev) / (2 * t_step)
            a = (u_prev - 2 * u + u_new) / (t_step ** 2)

            u_prev = u
            u = u_new
            force_previous = force_current

            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t2, u_new, v, a, K @ u_new, self.force.F)
                t2 += 1

        pbar.close()