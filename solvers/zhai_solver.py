import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolverInv
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, calculate_initial_acceleration, TimeIntegrationType


class ZhaiSolver(BaseSolverABC):
    """
    Zhai Solver class. This class contains the explicit solver according to :cite:p: `Zhai_1996`.
    """
    def __init__(self,
                 force: Force = Force(),
                 state: State = State(),
                 linear_solver: LinearSolversABC = SparseDirectSolverInv(),
                 preconditioner: PreconditionerABC = None,
                ):
        """
        Initialisation of the Zhai Solver class.

        Args:
            force (Force): Force class containing the force definitions (default: Force())
            state (State): State class containing the state (default: State())
            linear_solver (LinearSolversABC): Linear solver to be used (default: SparseDirectSolver())
            preconditioner (PreconditionerABC): Preconditioner to be used (default: None)
        """
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.force = force
        self.state = state
        self.psi = 0.5
        self.phi = 0.5
        self.beta = 1/4
        self.gamma = 1/2
        self.type = TimeIntegrationType.DYNAMIC

    def initialise(self, number_eq: int, time: npt.NDArray[np.float64]):
        """
        Initialise the solver state.

        Args:
            number_eq (int): Number of equations
            time (npt.NDArray[np.float64]): Time array
        """
        self.state.initialise(number_eq, time)

    def prediction(self, u, v, a, a_old, dt, is_initial):
        """
        Perform prediction for displacement and acceleration

        :param u: displacement
        :param v: velocity
        :param a: acceleration
        :param a_old: acceleration at previous time step
        :param dt: time step size
        :param is_initial: bool to indicate current iteration is the initial iteration
        :return:
        """

        # set Zhai factors
        if is_initial:
            psi = phi = 0
        else:
            psi = self.psi
            phi = self.phi

        # predict displacement and velocity
        u_new = u + v * dt + (1/2 + psi) * a * dt ** 2 - psi * a_old * dt**2
        v_new = v + (1 + phi) * a * dt - phi * a_old * dt
        return u_new, v_new

    def newmark_iteration(self, u: npt.NDArray[np.float64],
                          v: npt.NDArray[np.float64],
                          a: npt.NDArray[np.float64],
                          a_new: npt.NDArray[np.float64], dt: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Perform Newmark iteration as corrector for displacement and velocity

        Args:
            u (npt.NDArray[np.float64]): displacement
            v (npt.NDArray[np.float64]): velocity
            a (npt.NDArray[np.float64]): acceleration
            a_new (npt.NDArray[np.float64]): new acceleration
            dt (float): time step size
        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Newmark's displacement and velocity
        """
        u_new = u + v * dt + (1/2 - self.beta) * a * dt ** 2 + self.beta * a_new * dt ** 2
        v_new = v + (1-self.gamma) * a * dt + self.gamma * a_new * dt

        return u_new, v_new

    def calculate(self, M: Matrix, C: Matrix, K: Matrix, F: Matrix, t_start_idx: int, t_end_idx: int):
        """
        Perform calculation with the explicit Zhai solver

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

        # initial force conditions: for computation of initial acceleration
        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=self.state.u0)

        force = self.force.F

        # get initial displacement, velocity, acceleration and inverse mass matrix
        u = self.state.u0
        v = self.state.v0
        a = calculate_initial_acceleration(M, C, K, force, u, v, self.linear_solver, self.preconditioner)

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, v, a, force, self.force.F)

        a_old = np.zeros(self.state.number_equations)

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        is_initial = True
        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            # update force at time step
            self.force.update_rhs_at_time_step(t, u=u)

            # check if current timestep is the initial timestep
            if t > 1:
                is_initial = False

            # Predict displacement and velocity
            u_new, v_new = self.prediction(u, v, a, a_old, t_step, is_initial)

            # Calculate predicted external force vector
            self.force.update_rhs_at_non_linear_iteration(t, u=u_new)

            # Calculate predicted acceleration
            a_new = self.linear_solver.solve(M, self.force.F - K.dot(u_new) - C.dot(v_new))
            # Correct displacement and velocity
            u_new, v_new = self.newmark_iteration(u, v, a, a_new, t_step)

            # Calculate corrected force vector
            self.force.update_rhs_at_non_linear_iteration(t, u=u_new)

            # Calculate corrected acceleration
            a_new = self.linear_solver.solve(M, self.force.F - K.dot(u_new) - C.dot(v_new))

            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t, u_new, v_new, a_new, K @ u_new, self.force.F)
                t2 += 1

            # set vectors for next time step
            u = np.copy(u_new)
            v = np.copy(v_new)

            a_old = np.copy(a)
            a = np.copy(a_new)

        # close the progress bar
        pbar.close()
