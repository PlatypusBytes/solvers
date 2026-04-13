import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import Optional
from scipy.sparse import issparse as scipy_issparse

# Try to import CuPy and related sparse modules; if not available, set to None and handle in the class constructor
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
except ImportError:
    cp = None
    cpsp = None
    cpspla = None

from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolverLU
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType


class NewmarkImplicitForce(BaseSolverABC):
    """
    Implicit Newmark Solver class.
    """
    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None,
                 max_iter: int = 15,
                 tolerance: float = 1e-5
                 ):
        """
        Constructor of the Implicit Newmark Solver.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            linear_solver (Optional[LinearSolversABC]): Linear solver to be used (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner to be used (default: None)
            max_iter (int): Maximum number of iterations for the Newton-Raphson scheme (default: 15)
            tolerance (float): Tolerance for convergence in the Newton-Raphson scheme (default: 1e-5)
        """
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()
        self.max_iter = max_iter
        self.tolerance = tolerance

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
                self.state.store_step(t2, u, v, a, K @ u, self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()


class NewmarkExplicit(BaseSolverABC):
    """
    Explicit Newmark Solver class.
    """
    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None):
        """
        Constructor of the Explicit Newmark Solver.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            linear_solver (Optional[LinearSolversABC]): Linear solver to be used (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner to be used (default: None)
        """
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()

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

        # initialize force for the stage
        self.force.initialise_stage(F)
        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

        # update output arrays for the stage
        self.state.update_output_arrays(t_start_idx, t_end_idx)

        # check if sparse calculation should be performed
        M, C, K = self.state.check_for_sparse(M, C, K)

        # calculate time step size
        t_step = (self.state.time[t_end_idx] - self.state.time[t_start_idx]) / ((t_end_idx - t_start_idx))

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
                self.state.store_step(t2, u, v, a, K @ u, self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()



class NewmarkImplicitForceGPU(BaseSolverABC):
    """
    Implicit Newmark Solver class (GPU-accelerated with CuPy).

    The initial acceleration is computed on the CPU using ``linear_solver``.
    All time-stepping arithmetic is then performed on the GPU.
    Force callbacks (``Force`` class) remain CPU-side; their outputs are
    transferred to the GPU as needed, and results are moved back to CPU
    before being stored in ``State``.
    """
    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None,
                 max_iter: int = 15,
                 tolerance: float = 1e-5
                 ):
        """
        Constructor of the GPU-accelerated Implicit Newmark Solver.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            linear_solver (Optional[LinearSolversABC]): Linear solver used only for
                the initial acceleration computation on the CPU (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner for the
                CPU initial-acceleration solve (default: None); ignored on GPU
            max_iter (int): Maximum number of iterations for the Newton-Raphson scheme (default: 15)
            tolerance (float): Tolerance for convergence in the Newton-Raphson scheme (default: 1e-5)
        """
        if cp is None:
            raise ImportError(
                "CuPy is required for the GPU solver but is not installed. "
                "Install the package matching your CUDA version, e.g.:\n"
                "  pip install \"cupy-cuda12x[ctk]\"   # CUDA 12.x\n"
                "  pip install \"cupy-cuda13x[ctk]\"   # CUDA 13.x\n"
                "See https://docs.cupy.dev/en/stable/install.html"
            )
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()
        self.max_iter = max_iter
        self.tolerance = tolerance

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
        GPU-accelerated Newmark implicit integration scheme with Newton-Raphson
        strategy for non-linear force.  Incremental formulation.

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

        # constants for the Newmark integration - transfer to GPU
        beta = cp.array(self.beta)
        gamma = cp.array(self.gamma)
        t_step = cp.array(t_step)

        # CPU initial conditions
        u_cpu = self.state.u0
        v_cpu = self.state.v0

        # initial force conditions: for computation of initial acceleration (CPU)
        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=u_cpu)
        d_force_cpu = self.force.F

        # Send to GPU
        M_gpu = _to_gpu(M)
        C_gpu = _to_gpu(C)
        K_gpu = _to_gpu(K)
        d_force = cp.asarray(d_force_cpu, dtype=cp.float64)
        u = cp.asarray(u_cpu, dtype=cp.float64)
        v = cp.asarray(v_cpu, dtype=cp.float64)

        a = calculate_initial_acceleration(M_gpu, C_gpu, K_gpu, d_force, u, v,
                                           self.linear_solver, self.preconditioner)

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions (store CPU arrays directly)
        self.state.store_step(output_time_idx, u_cpu, v_cpu, a.get(), d_force_cpu, self.force.F)

        # combined stiffness matrix (assembled on GPU)
        K_till = K_gpu + C_gpu * (gamma / (beta * t_step)) + M_gpu * (1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        # F_previous is maintained on CPU for Force callbacks, and synced to GPU when needed
        F_previous_cpu = self.force.F.copy()

        # Build preconditioner if provided
        if self.preconditioner is not None:
            pre_c = self.preconditioner.build(K_till)
        else:
            pre_c = None

        for t in range(t_start_idx + 1, t_end_idx + 1):

            # update progress bar
            pbar.update(1)

            # get current displacement to CPU for force callbacks
            u_current = u.get()

            # update force at time step (CPU side; pass CPU copy of u)
            self.force.update_rhs_at_time_step(t, u=u_current)

            # updated mass contribution
            m_part = M_gpu.dot(v * (1 / (beta * t_step)) + a * (1 / (2 * beta)))
            # updated damping contribution
            c_part = C_gpu.dot(v * (gamma / beta) + a * (t_step * (gamma / (2 * beta) - 1)))

            # set ext force from previous time iteration (used for convergence check)
            force_ext_prev = d_force + m_part + c_part

            # initialise Newton-Raphson accumulators
            du_tot = cp.zeros_like(u)
            i = 0
            force_previous = 0
            F_previous_i_cpu = F_previous_cpu

            # Newton Raphson loop where force is updated in every iteration
            converged = False
            while not converged and i < self.max_iter:

                # update external force
                d_force_cpu_i, F_previous_i_cpu = self.force.update_force(u_current, F_previous_cpu, t)
                d_force = cp.asarray(d_force_cpu_i, dtype=cp.float64)

                # external force
                force_ext = d_force + m_part + c_part

                # solve on GPU
                du = self.linear_solver.solve(K_till, force_ext - force_previous, M=pre_c)

                # set du for first iteration
                if i == 0:
                    du_ini = cp.copy(du)

                # energy converge criterion according to bath 1996, chapter 8.4.4
                error = float(cp.linalg.norm(du * force_ext) / cp.linalg.norm(du_ini * force_ext_prev))
                converged = (error < self.tolerance)

                # calculate total du for current time step
                du_tot = du_tot + du

                # velocity calculated through Newmark relation
                dv = (
                        du_tot * (gamma / (beta * t_step))
                        - v * (gamma / beta)
                        + a * (t_step * (1 - gamma / (2 * beta)))
                )

                u = u + du

                if not converged:
                    force_previous = force_ext.copy()

                i += 1

            # acceleration calculated through Newmark relation
            da = (
                    du_tot * (1 / (beta * t_step ** 2))
                    - v * (1 / (beta * t_step))
                    - a * (1 / (2 * beta))
            )

            # update CPU force tracking for next time step
            F_previous_cpu = F_previous_i_cpu

            # update variables
            v = v + dv
            a = a + da

            # add to results (transfer GPU arrays back to CPU)
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t2, u.get(), v.get(), a.get(),
                                      (K_gpu @ u).get(), self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()


class NewmarkExplicitGPU(BaseSolverABC):
    """
    Explicit Newmark Solver class (GPU-accelerated with CuPy).

    The initial acceleration is computed on the CPU using ``linear_solver``.
    All time-stepping arithmetic is then performed on the GPU.
    """
    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None):
        """
        Constructor of the GPU-accelerated Explicit Newmark Solver.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            beta (float): Newmark numerical stability parameter (default: 0.25)
            gamma (float): Newmark numerical stability parameter (default: 0.5)
            linear_solver (Optional[LinearSolversABC]): Linear solver used only for
                the initial acceleration computation on the CPU (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner for the
                CPU initial-acceleration solve (default: None); ignored on GPU
        """
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()

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
        GPU-accelerated explicit Newmark integration scheme.
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
        t_step = (self.state.time[t_end_idx] - self.state.time[t_start_idx]) / ((t_end_idx - t_start_idx))

        # constants for the Newmark integration - transfer to GPU
        beta = cp.array(self.beta)
        gamma = cp.array(self.gamma)
        t_step = cp.array(t_step)

        # initial force conditions: for computation of initial acceleration (CPU)
        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=self.state.u0)
        d_force_cpu = self.force.F

        # CPU initial conditions
        u_cpu = self.state.u0
        v_cpu = self.state.v0

        # Send to GPU
        M_gpu = _to_gpu(M)
        C_gpu = _to_gpu(C)
        K_gpu = _to_gpu(K)
        d_force = cp.asarray(d_force_cpu, dtype=cp.float64)
        u = cp.asarray(u_cpu, dtype=cp.float64)
        v = cp.asarray(v_cpu, dtype=cp.float64)

        a = calculate_initial_acceleration(M_gpu, C_gpu, K_gpu, d_force, u, v,
                                           self.linear_solver, self.preconditioner)

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions (store CPU arrays directly)
        self.state.store_step(output_time_idx, u_cpu, v_cpu, a.get(), d_force_cpu, self.force.F)

        # combined stiffness matrix (assembled on GPU)
        K_till = K_gpu + C_gpu * (gamma / (beta * t_step)) + M_gpu * (1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        # F_previous is maintained on CPU for Force callbacks
        F_previous_cpu = self.force.F.copy()

        # Build preconditioner if provided
        if self.preconditioner is not None:
            pre_c = self.preconditioner.build(K_till)
        else:
            pre_c = None

        for t in range(t_start_idx + 1, t_end_idx + 1):

            # update progress bar
            pbar.update(1)

            # get current displacement to CPU for force callbacks
            u_current = u.get()

            # update force at time step (CPU side; pass CPU copy of u)
            self.force.update_rhs_at_time_step(t, u=u_current)

            # updated mass contribution
            m_part = M_gpu.dot(v * (1 / (beta * t_step)) + a * (1 / (2 * beta)))
            # updated damping contribution
            c_part = C_gpu.dot(v * (gamma / beta) + a * (t_step * (gamma / (2 * beta) - 1)))

            # update external force (CPU) and convert increment to GPU
            d_force_cpu, F_previous_cpu = self.force.update_force(u_current, F_previous_cpu, t)
            d_force = cp.asarray(d_force_cpu, dtype=cp.float64)

            # external force
            force_ext = d_force + m_part + c_part

            # solve on GPU
            du, _ = cpspla.cg(K_till, force_ext)
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

            # add to results (transfer GPU arrays back to CPU)
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t2, u.get(), v.get(), a.get(),
                                      (K_gpu @ u).get(), self.force.F)
                t2 += 1

        # close the progress bar
        pbar.close()


def _to_gpu(arr):
    """
    Transfer a CPU numpy array or scipy sparse matrix to the GPU.

    Args:
        arr: A numpy array or scipy sparse matrix.
    """
    if scipy_issparse(arr):
        return cpsp.csc_matrix(arr, dtype=cp.float64)
    return cp.asarray(arr, dtype=cp.float64)
