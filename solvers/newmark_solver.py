import sys
import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import solve, inv
from scipy.sparse.linalg import splu, spilu, LinearOperator, cg, spsolve
from tqdm import tqdm

from solvers.linear_equations_solvers import SolversABC, SparseDirectSolver
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import Force, State
from solvers.utils import eigen_decomposition

import numpy.typing as npt
import scipy.sparse as sp
from typing import Union, TypeAlias

# Define a custom type alias for readability
Matrix: TypeAlias = Union[npt.NDArray[np.float64], sp.spmatrix]

class NewmarkSolverABC(ABC):
    """
    Abstract base class for Newmark solvers.
    """
    @abstractmethod
    def calculate(self, M: Matrix, C: Matrix, K: Matrix, F: Matrix, t_start_idx: int, t_end_idx: int) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def u(self):
        """
        Dynamic accessor for displacement results from state.
        """
        return self.state.u

    @property
    def v(self):
        """
        Dynamic accessor for velocity results from state.
        """
        return self.state.v

    @property
    def a(self):
        """
        Dynamic accessor for acceleration results from state.
        """
        return self.state.a

    @property
    def time(self):
        """
        Dynamic accessor for time array from state.
        """
        return self.state.time

    @property
    def f(self):
        """
        Dynamic accessor for nodal force results from state.
        """
        return self.state.f


# class NewmarkSolver(Solver):
#     """
#     Newmark Solver class. This class contains the implicit incremental Newmark solver. This class bases from
#     :class:`~rose.model.solver.Solver`.

#     :Attributes:
#        - :self.beta:     Newmark numerical stability parameter
#        - :self.gamma:    Newmark numerical stability parameter
#     """

#     def __init__(self):
#         super(NewmarkSolver, self).__init__()
#         self.beta = 0.25
#         self.gamma = 0.5

#     def calculate_initial_acceleration(self, m_global, c_global, k_global, force_ini, u, v):
#         r"""
#         Calculation of the initial conditions - acceleration for the first time-step.

#         :param m_global: Global mass matrix
#         :param c_global: Global damping matrix
#         :param k_global: Global stiffness matrix
#         :param force_ini: Initial force
#         :param u: Initial conditions - displacement
#         :param v: Initial conditions - velocity

#         :return a: Initial acceleration
#         """

#         k_part = k_global.dot(u)
#         c_part = c_global.dot(v)

#         # initial acceleration
#         a, _ = self.solve_linear_system(m_global, force_ini - c_part - k_part, cache=False)
#         return a

#     def update_force(self, u, F_previous, t):
#         """
#         Updates the external force vector at time t

#         :param u: displacement vector at time t
#         :param F_previous: Force vector at previous time step
#         :param t:  current time step index
#         :return:
#         """

#         # calculates force with custom load function
#         self.update_rhs_at_non_linear_iteration(t,u=u)

#         force = self.F

#         # calculate force increment with respect to the previous time step
#         d_force = force - F_previous

#         # copy force vector such that force vector data at each time step is maintained
#         F_total = np.copy(force)

#         return d_force, F_total

#     def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
#         """
#         Base calculation function of the Newmark Solver. This function does not do any calculation, instead an error
#         message is returned that any of the inherited Newmark solvers should be used.

#         :param M: Mass matrix
#         :param C: Damping matrix
#         :param K: Stiffness matrix
#         :param F: External force matrix
#         :param t_start_idx: time index of starting time for the stage analysis
#         :param t_end_idx: time index of end time for the stage analysis
#         :return:
#         """
#         logging.error("Calculate function of the base NewmarkSolver is called. "
#                       "Use 'NewmarkImplicitForce' or 'NewmarkExplicit' instead")


class NewmarkImplicitForce(NewmarkSolverABC):

    def __init__(self):
        # super(NewmarkImplicitForce, self).__init__()
        self.max_iter = 15
        self.tolerance = 1e-5

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Newmark implicit integration scheme with Newton Raphson strategy for non-linear force.
        Incremental formulation.

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
        """

        self.initialise_stage(F)

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        self.update_output_arrays(t_start_idx, t_end_idx)
        # validate solver index
        self.validate_input(t_start_idx, t_end_idx)

        # calculate time step size
        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        du = np.zeros(len(u))

        self.update_rhs_at_time_step(t_start_idx)
        self.update_rhs_at_non_linear_iteration(t_start_idx,u=u)

        # initial force conditions: for computation of initial acceleration

        d_force = self.F

        a = self.calculate_initial_acceleration(M, C, K, d_force, u, v)

        # initialise delta velocity
        dv = np.zeros(len(v))

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a
        self.f[output_time_idx, :] = d_force
        self.F_out[output_time_idx, :] = np.copy(self.F)

        # combined stiffness matrix
        K_till = K + C * (gamma / (beta * t_step)) + M * (1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        # initialise Force from load function
        F_previous = np.copy(self.F)

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):
            self.update_rhs_at_time_step(t, u=u)

            # update progress bar
            pbar.update(1)

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
                d_force, F_previous_i = self.update_force(u, F_previous, t)

                # external force
                force_ext = d_force + m_part + c_part

                # # solve
                # if self._is_sparse_calculation:
                #     pre_conditioner = True
                #     if pre_conditioner:
                #         diagonal = K_till.diagonal()
                #         M_op = LinearOperator(shape=K_till.shape, matvec=lambda v: v / diagonal)

                #         du, info = cg(K_till, force_ext - force_previous, x0=du, M=M_op, rtol=1e-12, maxiter=10_000)
                #         if info > 0:
                #             sys.error(f"ERROR: not converged time step {t}")
                #     else:
                #         inv_K_till = splu(K_till)
                # else:
                #     inv_K_till = inv(K_till)
                du, info = self.solve_linear_system(K_till, force_ext - force_previous)
                if info > 0:
                    sys.error(f"ERROR: not converged time step {t}")

                # if self._is_sparse_calculation:
                #     du = inv_K_till.solve(force_ext - force_previous)
                # else:
                #     du = inv_K_till.dot(force_ext - force_previous)

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
            if t == self.output_time_indices[t2]:
                self.u[t2, :] = u
                self.v[t2, :] = v
                self.a[t2, :] = a

                self.F_out[t2, :] = np.copy(self.F)
                t2 += 1

        # calculate nodal force
        self.f[:, :] = np.transpose(K.dot(np.transpose(self.u)))
        # clear the cached inverse of K_till for multistage analyses
        self.cache_inv_K_till = None
        # close the progress bar
        pbar.close()


class NewmarkExplicit(NewmarkSolverABC):
    """
    Explicit Newmark Solver class.
    """
    def __init__(self,
                 force: Force,
                 state: State,
                 beta: float = 0.25,
                 gamma: float = 0.5,
                 linear_solver: SolversABC = SparseDirectSolver,
                 preconditioner: PreconditionerABC = None):
        """
        Constructor of the Explicit Newmark Solver.

        Parameters
        ----------
        :param force: Force class containing the force definitions
        :param state: State class containing the state
        :param beta: Newmark numerical stability parameter
        :param gamma: Newmark numerical stability parameter
        :param linear_solver: Linear solver to be used (default: SparseDirectSolver)
        :param preconditioner: Preconditioner to be used (default: None)
        """
        self.beta = beta
        self.gamma = gamma
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.force = force
        self.state = state

    def initialise(self, number_eq: int, time: np.ndarray):
        """
        Initialise the solver state.

        Parameters
        ----------
        :param number_eq: Number of equations
        :param time: Time array
        """
        self.state.initialise(number_eq, time)

    def calculate(self, M: Matrix, C: Matrix, K: Matrix, F: Matrix, t_start_idx: int, t_end_idx: int):
        """
        Explicit newmark integration scheme.
        Incremental formulation.

        Parameters
        ----------
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
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
        self.state.store_step(t_start_idx, u, v, a, d_force, self.force.F)

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

class ModalAnalysisNewmark(NewmarkSolverABC):

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
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


        self.initialise_stage(F)

        # check if sparse calculation should be performed
        M, C, K = self.check_for_sparse(M, C, K)

        self.update_output_arrays(t_start_idx, t_end_idx)
        # validate solver index
        self.validate_input(t_start_idx, t_end_idx)

        # calculate time step size
        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial force conditions: for computation of initial acceleration
        self.update_rhs_at_time_step(t_start_idx)
        self.update_rhs_at_non_linear_iteration(t_start_idx, u=self.u0)

        d_force = self.F

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        a = self.calculate_initial_acceleration(M, C, K, d_force, u, v)

        # initialise delta velocity
        dv = np.zeros(len(v))

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.u[output_time_idx, :] = u
        self.v[output_time_idx, :] = v
        self.a[output_time_idx, :] = a
        self.f[output_time_idx, :] = d_force

        self.F_out[output_time_idx, :] = np.copy(self.F)

        eigvals, eigvecs = eigen_decomposition(M, K)
        # # eigen-decomposition (dense) -> ensure use of dense arrays for eigh
        # eigvals, eigvecs = eigh(K.todense(), M.todense())
        omega_values = np.sqrt(np.maximum(eigvals, 0.0))
        frequencies = omega_values / (2.0 * np.pi)
        idx_freq = frequencies <= 100
        frequencies = frequencies[idx_freq]
        eigen_vectors = np.asarray(eigvecs[:, idx_freq])
        eigen_vectors = eigvecs[:, idx_freq]

        # modal mass
        M_modal = np.diag(eigen_vectors.T @ M.todense() @ eigen_vectors)
        # modal damping
        C_modal = np.diag(eigen_vectors.T @ C.todense() @ eigen_vectors)
        # modal stiffness
        K_modal = np.diag(eigen_vectors.T @ K.todense() @ eigen_vectors)

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
        F_previous = np.copy(self.F)

        # Project initial physical variables into modal coords (1D arrays)
        qu = eigen_vectors.T @ u
        qv = eigen_vectors.T @ v
        qa = eigen_vectors.T @ a

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):

            self.update_rhs_at_time_step(t, u=u)

            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = qv * (1 / (beta * t_step)) + qa * (1 / (2 * beta))
            m_part = M_modal * m_part
            # updated damping
            c_part = qv * (gamma / beta) + qa * (t_step * (gamma / (2 * beta) - 1))
            c_part = C_modal * c_part


            # update external force
            d_force, F_previous = self.update_force(u, F_previous, t)
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

            # add to results
            if t == self.output_time_indices[t2]:
                v = eigen_vectors @ qv
                a = eigen_vectors @ qa

                self.u[t2, :] = u
                self.v[t2, :] = v
                self.a[t2, :] = a

                self.F_out[t2, :] = np.copy(self.F)
                t2 += 1


def calculate_initial_acceleration(m_global: Matrix,
                                   c_global: Matrix,
                                   k_global: Matrix,
                                   force_ini: npt.NDArray[np.float64],
                                   u: npt.NDArray[np.float64],
                                   v: npt.NDArray[np.float64],
                                   linear_solver: SolversABC,
                                   preconditioner: PreconditionerABC) -> npt.NDArray[np.float64]:
    r"""
    Calculation of the initial conditions - acceleration for the first time-step.

    Args:
        m_global (Matrix): Global mass matrix
        c_global (Matrix): Global damping matrix
        k_global (Matrix): Global stiffness matrix
        force_ini (npt.NDArray[np.float64]): Initial force
        u (npt.NDArray[np.float64]): Initial conditions - displacement
        v (npt.NDArray[np.float64]): Initial conditions - velocity
        linear_solver (SolversABC): Linear solver instance to solve the linear system
        preconditioner (PreconditionerABC): Preconditioner instance to be used in the linear solver
    Returns:
        a (npt.NDArray[np.float64]): Initial acceleration
    """

    k_part = k_global.dot(u)
    c_part = c_global.dot(v)

    if preconditioner is not None:
        pre_c = preconditioner.build(m_global)
    else:
        pre_c = None

    # initial acceleration
    a = linear_solver.solve(m_global, force_ini - c_part - k_part, M=pre_c)
    return a
