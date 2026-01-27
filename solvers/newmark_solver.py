import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolver
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType


class NewmarkImplicitForce(BaseSolverABC):
    """
    Implicit Newmark Solver class.
    """
    def __init__(self,
                 force: Force = Force(),
                 state: State = State(),
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: LinearSolversABC = SparseDirectSolver(),
                 preconditioner: PreconditionerABC = None,
                 max_iter: int = 15,
                 tolerance: float = 1e-5
                 ):
        """
        Constructor of the Implicit Newmark Solver.

        Args:
            force (Force): Force class containing the force definitions (default: Force())
            state (State): State class containing the state (default: State())
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            linear_solver (LinearSolversABC): Linear solver to be used (default: SparseDirectSolver())
            preconditioner (PreconditionerABC): Preconditioner to be used (default: None)
            max_iter (int): Maximum number of iterations for the Newton-Raphson scheme (default: 15)
            tolerance (float): Tolerance for convergence in the Newton-Raphson scheme (default: 1e-5)
        """
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.force = force
        self.state = state
        self.max_iter = max_iter
        self.tolerance = tolerance
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
        Newmark implicit integration scheme with Newton Raphson strategy for non-linear force.
        Incremental formulation.

        Args:
            M (Matrix): Mass matrix
            C (Matrix): Damping matrix
            K (Matrix): Stiffness matrix
            F (Matrix): External force matrix
            t_start_idx (int): time index of starting time for the stage analysis
            t_end_idx (int): time index of end time for the stage analysis
        """

        self.force.initialise_stage(F)

        # check if sparse calculation should be performed
        M, C, K = self.state.check_for_sparse(M, C, K)

        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

        # calculate time step size
        t_step = (self.state.time[t_end_idx] - self.state.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial conditions u, v, a
        u = self.state.u0
        v = self.state.v0
        du = np.zeros(len(u))

        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=u)

        # initial force conditions: for computation of initial acceleration
        d_force = self.force.F
        a = calculate_initial_acceleration(M, C, K, d_force, u, v, self.linear_solver, self.preconditioner)

        # initialise delta velocity
        dv = np.zeros(len(v))

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, v, a, d_force, self.force.F)

        # combined stiffness matrix
        K_till = K + C * (gamma / (beta * t_step)) + M * (1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        # initialise Force from load function
        F_previous = np.copy(self.force.F)

        # Build preconditioner if provided
        if self.preconditioner is not None:
            pre_c = self.preconditioner.build(K_till)
        else:
            pre_c = None

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):

            # update progress bar
            pbar.update(1)

            # update force at time step
            self.force.update_rhs_at_time_step(t, u=u)

            # updated mass
            m_part = v * (1 / (beta * t_step)) + a * (1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v * (gamma / beta) + a * (t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # set ext force from previous time iteration
            force_ext_prev = d_force + m_part + c_part

            # initialise
            du_tot = 0
            i = 0
            force_previous = 0

            # Newton Raphson loop where force is updated in every iteration
            converged = False
            while not converged and i < self.max_iter:

                # update external force
                d_force, F_previous_i = self.force.update_force(u, F_previous, t)

                # external force
                force_ext = d_force + m_part + c_part

                # solve
                du = self.linear_solver.solve(K_till, force_ext - force_previous, M=pre_c)

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
                        du_tot * (gamma / (beta * t_step))
                        - v * (gamma / beta)
                        + a * (t_step * (1 - gamma / (2 * beta)))
                )

                u = u + du

                if not converged:
                    force_previous = np.copy(force_ext)

                i += 1

            # acceleration calculated through Newmark relation
            da = (
                    du_tot * (1 / (beta * t_step ** 2))
                    - v * (1 / (beta * t_step))
                    - a * (1 / (2 * beta))
            )

            F_previous = F_previous_i
            # update variables

            v = v + dv
            a = a + da

            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t, u, v, a, K @ u, self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()


class NewmarkExplicit(BaseSolverABC):
    """
    Explicit Newmark Solver class.
    """
    def __init__(self,
                 force: Force = Force(),
                 state: State = State(),
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: LinearSolversABC = SparseDirectSolver(),
                 preconditioner: PreconditionerABC = None):
        """
        Constructor of the Explicit Newmark Solver.

        Args:
            force (Force): Force class containing the force definitions (default: Force())
            state (State): State class containing the state (default: State())
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            linear_solver (LinearSolversABC): Linear solver to be used (default: SparseDirectSolver())
            preconditioner (PreconditionerABC): Preconditioner to be used (default: None)
        """
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.force = force
        self.state = state
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
        Explicit newmark integration scheme.
        Incremental formulation.

        Args:
            M (Matrix): Mass matrix
            C (Matrix): Damping matrix
            K (Matrix): Stiffness matrix
            F (Matrix): External force matrix
            t_start_idx (int): time index of starting time for the stage analysis
            t_end_idx (int): time index of end time for the stage analysis
        """
        self.force.initialise_stage(F)

        # check if sparse calculation should be performed
        M, C, K = self.state.check_for_sparse(M, C, K)

        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

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

        # initialise delta velocity
        dv = np.zeros(len(v))

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, v, a, d_force, self.force.F)

        # combined stiffness matrix
        K_till = K + C * (gamma / (beta * t_step)) + M * (1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        # initialise Force from load function
        F_previous = np.copy(self.force.F)

        # Build preconditioner if provided
        if self.preconditioner is not None:
            pre_c = self.preconditioner.build(K_till)
        else:
            pre_c = None

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):

            # update progress bar
            pbar.update(1)

            # update force at time step
            self.force.update_rhs_at_time_step(t, u=u)

            # updated mass
            m_part = v * (1 / (beta * t_step)) + a * (1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v * (gamma / beta) + a * (t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # update external force
            d_force, F_previous = self.force.update_force(u, F_previous, t)

            # external force
            force_ext = d_force + m_part + c_part

            # solve
            du = self.linear_solver.solve(K_till, force_ext, M=pre_c)

            # velocity calculated through Newmark relation
            dv = (
                du * (gamma / (beta * t_step))
                - v * (gamma / beta)
                + a * (t_step * (1 - gamma / (2 * beta)))
            )

            # acceleration calculated through Newmark relation
            da = (
                du * (1 / (beta * t_step ** 2))
                - v * (1 / (beta * t_step))
                - a * (1 / (2 * beta))
            )

            # update variables
            u = u + du
            v = v + dv
            a = a + da

            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t, u, v, a, K @ u, self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()
