from tqdm import tqdm
import numpy as np
import numpy.typing as npt

from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolver
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType


class HHTImplicitForce(BaseSolverABC):
    """
    Hilber-Hughes-Taylor implicit integration scheme with Newton Raphson strategy for non-linear force.
    """
    def __init__(self,
                 force: Force = Force(),
                 state: State = State(),
                 linear_solver: LinearSolversABC = SparseDirectSolver(),
                 preconditioner: PreconditionerABC = None,
                 alpha: float = 1/6,
                 max_iter: int =15,
                 tolerance: float = 1e-5):
        """
        Initializes the HHT implicit solver with given parameters.

        Args:
            force (Force): Force instance (default: Force())
            state (State): State instance (default: State())
            linear_solver (LinearSolversABC): Linear solver instance (default: SparseDirectSolver())
            preconditioner (PreconditionerABC): Preconditioner instance (default: None)
            alpha (float): HHT alpha parameter for numerical dissipation.
            max_iter (int): Maximum number of iterations for the Newton-Raphson method.
            tolerance (float): Convergence tolerance for the Newton-Raphson method.
        """
        self.force = force
        self.state = state
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.alpha = alpha
        self.type = TimeIntegrationType.DYNAMIC

        self.beta = (1 + self.alpha)**2/4
        self.gamma = (1 + 2 * self.alpha)/2
        self.max_iter = max_iter
        self.tolerance = tolerance

    def initialise(self, number_eq: int, time: np.ndarray):
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
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        # initial conditions u, v, a
        u = self.state.u0
        v = self.state.v0

        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx,u=u)

        # initial force conditions: for computation of initial acceleration
        d_force = self.force.F
        a = calculate_initial_acceleration(M, C, K, d_force, u, v, self.linear_solver, self.preconditioner)

        # initialise delta velocity
        dv = np.zeros(len(v))

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        self.state.store_step(output_time_idx, u, v, a, d_force, self.force.F)

        # combined stiffness matrix
        K_till = K*(1-alpha) + C * ((gamma*(1-alpha)) / (beta * t_step)) + M * (1 / (beta * t_step ** 2))

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

            # update force at current time step
            self.force.update_rhs_at_time_step(t, u=u)

            # updated mass
            m_part = v * (1 / (beta * t_step)) + a * (1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = (1 - alpha) * (v * (gamma / beta) + a * (t_step * (gamma / (2 * beta) - 1)))
            c_part = C.dot(c_part)

            # set ext force from previous time iteration
            force_ext_prev = d_force*(1-alpha) + m_part + c_part

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
                force_ext = d_force*(1-alpha) + m_part + c_part

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


class HHTExplicit(BaseSolverABC):
    """
    Hilber-Hughes-Taylor explicit integration scheme.
    """
    def __init__(self,
                 force: Force = Force(),
                 state: State = State(),
                 linear_solver: LinearSolversABC = SparseDirectSolver(),
                 preconditioner: PreconditionerABC = None,
                 alpha: float = 1/6,
                 ):
        """
        Initializes the HHT explicit solver with given parameters.

        Args:
            force (Force): Force instance (default: Force())
            state (State): State instance (default: State())
            linear_solver (LinearSolversABC): Linear solver instance (default: SparseDirectSolver())
            preconditioner (PreconditionerABC): Preconditioner instance (default: None)
            alpha (float): HHT alpha parameter for numerical dissipation.
        """
        self.force = force
        self.state = state
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.alpha = alpha
        self.type = TimeIntegrationType.DYNAMIC

        self.beta = (1 + self.alpha)**2/4
        self.gamma = (1 + 2 * self.alpha)/2

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

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
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
        alpha = self.alpha
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

        self.state.store_step(output_time_idx, u, v, a, d_force, self.force.F)

        # combined stiffness matrix
        K_till = K*(1-alpha) + C * ((gamma*(1-alpha)) / (beta * t_step)) + M * (1 / (beta * t_step ** 2))

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

            # update force at current time step
            self.force.update_rhs_at_time_step(t, u=u)

            # updated mass
            m_part = v * (1 / (beta * t_step)) + a * (1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = (1-alpha)*(v * (gamma / beta) + a * (t_step * (gamma / (2 * beta) - 1)))
            c_part = C.dot(c_part)

            # update external force
            d_force, F_previous = self.force.update_force(u, F_previous, t)

            # external force
            force_ext = d_force*(1-alpha) + m_part + c_part

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
