import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from solvers.utils import LumpingMethod
from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolverInv
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType


class BatheSolver(BaseSolverABC):
    """
    Bathe Solver class.
    This class contains the explicit solver according to :cite:p: `Noh_Bathe_2013`.
    """

    def __init__(self,
                 force: Force = Force(),
                 state: State = State(),
                 linear_solver: LinearSolversABC = SparseDirectSolverInv(),
                 preconditioner: PreconditionerABC = None,
                 lumping_method=LumpingMethod.RowSum
                 ):
        """
        Initialisation of the Bathe Solver class.

        Args:
            force (Force): Force class containing the force definitions (default: Force())
            state (State): State class containing the state (default: State())
            linear_solver (LinearSolversABC): Linear solver to be used (default: SparseDirectSolver())
            preconditioner (PreconditionerABC): Preconditioner to be used (default: None)
            lumping_method (LumpingMethod): Lumping method to be used (default: RowSum)
        """
        if not isinstance(lumping_method, LumpingMethod):
            raise ValueError("Lumping method must be of type LumpingMethod")

        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.force = force
        self.state = state
        self.lump_method = lumping_method
        self.is_lumped = lumping_method != LumpingMethod.NONE
        self._p = 0.54  # Bathe parameter
        self.type = TimeIntegrationType.DYNAMIC

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
        Perform calculation with the Bathe solver.

        Parameters:
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        """

        # initialize force for the stage
        self.force.initialise_stage(F)
        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

        # update output arrays for the stage
        self.state.update_output_arrays(t_start_idx, t_end_idx)

        # check if sparse calculation should be performed
        M, C, K = self.state.check_for_sparse(M, C, K)

        # calculate step size
        t_step = (self.state.time[t_end_idx] - self.state.time[t_start_idx]) / (t_end_idx - t_start_idx)

        # initial force conditions: for computation of initial acceleration
        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=self.state.u0)

        d_force = self.force.F

        if self.is_lumped:
            # Lumping only applies to the mass matrix the damping matrix remain consistent
            M_diag = self.lump_method.apply(M)
            inv_M_diag = 1 / M_diag

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
        u = self.state.u0
        v = self.state.v0
        if self.is_lumped:
            a = inv_M_diag * (self.force.F - K.dot(u) - C.dot(v))
        else:
            a = calculate_initial_acceleration(M, C, K, d_force, u, v, self.linear_solver, self.preconditioner)

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, v, a, d_force, self.force.F)

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        # initialise Force from load function
        force_previous = np.copy(self.force.F)

        # Build preconditioner if provided
        if self.preconditioner is not None:
            pre_c = self.preconditioner.build(M)
        else:
            pre_c = None

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            self.force.update_rhs_at_time_step(t, u=u)

            # update external force
            _, force = self.force.update_force(u, force_previous, t)

            # first sub-step
            u_t_p = u + a0 * v + a1 * a
            force_term = (1 - self._p) * force_previous + self._p * force
            force_term = force_term - K.dot(u_t_p) - C.dot(v + a0 * a)

            if self.is_lumped:
                a_t_p = inv_M_diag * force_term
            else:
                a_t_p = self.linear_solver.solve(M, force_term, M=pre_c)

            v_t_p = v + a2 * (a + a_t_p)

            # second sub-step
            u = u_t_p + a3 * v_t_p + a4 * a_t_p
            force_term = force - K.dot(u) - C.dot(v_t_p + a3 * a_t_p)
            if self.is_lumped:
                a_next = inv_M_diag * force_term
            else:
                a_next = self.linear_solver.solve(M, force_term, M=pre_c)
            v = v_t_p + a5 * a + a6 * a_t_p + a7 * a_next
            a = a_next
            force_previous = np.copy(force)

            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t, u, v, a, K @ u, self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()
