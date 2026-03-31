from typing import Optional
import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolverLU
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType


class ModalNewmark(BaseSolverABC):

    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 max_modes: int = 1000,
                 frequency_cutoff_hz: float = 100.0,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None,
                 max_iter: int = 15,
                 tolerance: float = 1e-5,
                 ):
        """
        Constructor of the Implicit Newmark Solver.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            max_modes (int): Maximum number of modes extracted for sparse eigenvalue analysis (default: 200)
            frequency_cutoff_hz (float): Frequency cutoff used for modal truncation (default: 100.0)
            linear_solver (Optional[LinearSolversABC]): Linear solver to be used (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner to be used (default: None)
            max_iter (int): Maximum number of iterations for the Newton-Raphson scheme (default: 15)
            tolerance (float): Tolerance for convergence in the Newton-Raphson scheme (default: 1e-5)
        """
        self.beta = beta
        self.gamma = gamma
        self.max_modes = max_modes
        self.frequency_cutoff_hz = frequency_cutoff_hz
        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()
        self.max_iter = max_iter
        self.tolerance = tolerance

    def __extract_modes(self, M: Matrix, C: Matrix, K: Matrix) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Extracts vibration modes and returns diagonal modal matrices.

        Returns:
            tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
            ]: modal eigenvectors, modal mass, damping and stiffness diagonals.
        """
        number_eq = M.shape[0]

        if (issparse(M) or issparse(K)):
            max_k = min(self.max_modes, number_eq - 1)
            eigvals, eigvecs = eigsh(K, k=max_k, M=M, which="SM")
        else:
            eigvals, eigvecs = eigh(np.asarray(K), np.asarray(M))

        omega_values = np.sqrt(eigvals)
        frequencies = omega_values / (2.0 * np.pi)
        idx_freq = frequencies <= self.frequency_cutoff_hz

        # check if any modes are selected, if not raise an error
        if not np.any(idx_freq):
            raise ValueError("No vibration modes selected. Increase the frequency cutoff.")

        # check if frequency cutoff is reached before max modes, if not raise a warning
        if np.sum(idx_freq) == self.max_modes:
            raise Warning(
                f"Maximum number of modes ({self.max_modes}) reached before frequency cutoff.\n"
                "Consider increasing max_modes or decreasing frequency_cutoff_hz."
            )

        eigen_vectors = np.asarray(eigvecs[:, idx_freq])

        # Products keep sparsity where available and produce dense modal operators.
        M_modal = np.diag(eigen_vectors.T @ (M @ eigen_vectors))
        C_modal = np.diag(eigen_vectors.T @ (C @ eigen_vectors))
        K_modal = np.diag(eigen_vectors.T @ (K @ eigen_vectors))

        return eigen_vectors, M_modal, C_modal, K_modal

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
        Initialise the solver state.

        Args:
            number_eq (int): Number of equations
            time (npt.NDArray[np.float64]): Time array
        """
        self.state.initialise(number_eq, time)


    def calculate(self, M: Matrix, C: Matrix, K: Matrix, F: Matrix, t_start_idx: int, t_end_idx: int):
        """
        Newmark integration scheme.
        Incremental formulation.

        Args:
            M (Matrix): Mass matrix
            C (Matrix): Damping matrix
            K (Matrix): Stiffness matrix
            F (Matrix): External force matrix
            t_start_idx (int): time index of starting time for the stage analysis
            t_end_idx (int): time index of end time for the stage analysis
        """

        # initialize force for the stage
        self.force.initialise_stage(F)
        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

        # update output arrays for the stage
        self.state.update_output_arrays(t_start_idx, t_end_idx)

        # check if sparse calculation should be performed
        M, C, K = self.state.check_for_sparse(M, C, K)

        # calculate time step size
        t_step = (self.state.time[t_end_idx] - self.state.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial force conditions: for computation of initial acceleration
        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=self.state.u0)

        d_force = self.force.F

        # initial conditions u, v, a
        u = self.state.u0
        v = self.state.v0
        a = calculate_initial_acceleration(M, C, K, d_force, u, v, self.linear_solver, self.preconditioner)

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, v, a, d_force, self.force.F)

        # modal decomposition
        eigen_vectors, M_modal, C_modal, K_modal = self.__extract_modes(M, C, K)

        # combined stiffness matrix
        K_till = K_modal + C_modal * (gamma / (beta * t_step)) + M_modal * (1 / (beta * t_step ** 2))
        inv_K_till = 1 / K_till

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        # initialise Force from load function
        F_previous = np.copy(self.force.F)

        # Project initial physical variables into modal coords (1D arrays)
        qu = eigen_vectors.T @ u
        qv = eigen_vectors.T @ v
        qa = eigen_vectors.T @ a

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):

            # update progress bar
            pbar.update(1)

            # update force at time step
            self.force.update_rhs_at_time_step(t, u=u)

            # updated mass
            m_part = qv * (1 / (beta * t_step)) + qa * (1 / (2 * beta))
            m_part = M_modal * m_part
            # updated damping
            c_part = qv * (gamma / beta) + qa * (t_step * (gamma / (2 * beta) - 1))
            c_part = C_modal * c_part


            # update external force
            d_force, F_previous = self.force.update_force(u, F_previous, t)
            d_force_modal = eigen_vectors.T @ d_force

            # external force
            force_ext = d_force_modal + m_part + c_part

            # solve
            dqu = inv_K_till * force_ext

            # velocity calculated through Newmark relation
            dqv = (
                dqu * (gamma / (beta * t_step))
                - qv * (gamma / beta)
                + qa * (t_step * (1 - gamma / (2 * beta)))
            )

            # acceleration calculated through Newmark relation
            dqa = (
                dqu * (1 / (beta * t_step ** 2))
                - qv * (1 / (beta * t_step))
                - qa * (1 / (2 * beta))
            )

            # update variables
            qu = qu + dqu
            qv = qv + dqv
            qa = qa + dqa

            # reconstruct physical displacements, velocities and accelerations
            u = eigen_vectors @ qu
            v = eigen_vectors @ qv
            a = eigen_vectors @ qa

            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t2, u, v, a, K @ u, self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()
