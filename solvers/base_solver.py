from abc import ABC, abstractmethod
from typing import Union, Optional, TypeAlias
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import issparse, csc_matrix

from solvers.linear_equations_solvers import LinearSolversABC
from solvers.preconditioners import PreconditionerABC


# Define a custom type alias for readability
Matrix: TypeAlias = Union[npt.NDArray[np.float64], sp.spmatrix]


class TimeIntegrationType(Enum):
    """
    Enum for time integration types.
    """
    STATIC = auto()
    DYNAMIC = auto()

class BaseSolverABC(ABC):
    """
    Abstract base class for Newmark solvers.
    """
    @abstractmethod
    def calculate(self, *args):
        """
        Abstract method to perform the calculation of the solver.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def u(self):
        """
        Dynamic accessor for displacement results from state.

        Returns:
            npt.NDArray[np.float64]: Displacement results from state.
        """
        return self.state.u

    @property
    def v(self):
        """
        Dynamic accessor for velocity results from state.

        Returns:
            npt.NDArray[np.float64]: Velocity results from state.
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
        Dynamic accessor for the output time array from state.

        Returns:
            npt.NDArray[np.float64]: Output time array from state.
        """
        return self.state.output_time

    @property
    def f(self):
        """
        Dynamic accessor for nodal force results from state.

        Returns:
            npt.NDArray[np.float64]: Nodal force results from state.
        """
        return self.state.f

class State:
    """
    State class. This class forms the base for each solver.
    """
    def __init__(self, output_interval: int = 1):
        """
        Initializes the State class with default attributes.
        """
        self.u0 = None
        self.v0 = None
        self.u = None
        self.v = None
        self.a = None
        self.f = None
        self.time = None
        self.output_interval = output_interval
        self.F_out = None
        self.output_time = None
        self.output_time_indices = None
        self.number_equations = None


    def initialise(self, number_equations: int, time: npt.NDArray[np.float64]):
        """
        Initializes displacement and velocity vectors
        Initializes output time interval and output matrices

        Args:
            number_equations (int): Number of equations to be solved.
            time (npt.NDArray[np.float64]): Time discretisation.
        """
        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

        self.number_equations = number_equations
        self.time = np.array(time)

        # find indices of time steps which should be stored based on output interval
        self.output_time_indices = np.arange(0, len(self.time), self.output_interval)

        # make sure last time step is included
        if not np.isclose(self.output_time_indices[-1], len(self.time) - 1):
            self.output_time_indices = np.append(self.output_time_indices, len(self.time) - 1)

        # initialise result arrays
        self.output_time = self.time[self.output_time_indices]
        self.u = np.zeros((len(self.output_time_indices), number_equations))
        self.v = np.zeros((len(self.output_time_indices), number_equations))
        self.a = np.zeros((len(self.output_time_indices), number_equations))
        self.f = np.zeros((len(self.output_time_indices), number_equations))

        self.F_out = np.zeros((len(self.output_time_indices), number_equations))

    def check_for_sparse(self, M: Matrix, C: Matrix, K: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        """
        Checks if one of the input matrices is a sparse matrix.
        If so, convert all input matrices to csc sparse matrices.

        Args:
            M (Matrix): Mass matrix.
            C (Matrix): Damping matrix.
            K (Matrix): Stiffness matrix.

        Returns:
            tuple[Matrix, Matrix, Matrix]: Possibly converted mass, damping, and stiffness matrices.
        """
        # check if sparse calculation should be performed
        if issparse(M) or issparse(C) or issparse(K):
            M = csc_matrix(M)
            C = csc_matrix(C)
            K = csc_matrix(K)

        return M, C, K

    def store_step(self, t_index: int, u: npt.NDArray[np.float64], v: npt.NDArray[np.float64],
                   a: npt.NDArray[np.float64], f: npt.NDArray[np.float64], F: npt.NDArray[np.float64]):
        """
        Store results at time index t_index if it is an output time.

        Args:
            t_index (int): Time index.
            u (npt.NDArray[np.float64]): Displacement vector at time t_index.
            v (npt.NDArray[np.float64]): Velocity vector at time t_index.
            a (npt.NDArray[np.float64]): Acceleration vector at time t_index.
            f (npt.NDArray[np.float64]): Internal force vector at time t_index.
            F (npt.NDArray[np.float64]): External force vector at time t_index
        """

        self.u[t_index] = u
        self.v[t_index] = v
        self.a[t_index] = a
        self.f[t_index] = f
        self.F_out[t_index] = F

    def update_initial_conditions(self, t_start_idx: int):
        """
        Updates the initial conditions on a certain stage.
        Initial conditions are retrieved from previously calculated values for
        displacements and velocities.

        Args:
            t_start_idx (int): start time index of current stage
        """
        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]

        self.u0 = self.u[output_time_idx, :]
        self.v0 = self.v[output_time_idx, :]

    def update_output_arrays(self, t_start_idx: int, t_end_idx: int):
        """
        Updates output arrays.
        If either the t_start_idx or t_end_idx is missing in the output indices array, these indices are added.

        Args:
            t_start_idx (int): start time index of current stage
            t_end_idx (int): end time index of current stage
        """

        # add start time index if required
        if t_start_idx not in self.output_time_indices:
            closest_greater_index = np.where(self.output_time_indices[self.output_time_indices >t_start_idx].min() == self.output_time_indices)[0]
            self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_start_idx)
            self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
            self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
            self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
            self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)
            self.F_out = np.insert(self.F_out, closest_greater_index, np.zeros(self.F_out.shape[1]), axis=0)
            self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_start_idx])

        # add end time index if required
        if t_end_idx not in self.output_time_indices:
            closest_greater_index = np.where(self.output_time_indices[self.output_time_indices >t_end_idx].min() == self.output_time_indices)[0]
            self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_end_idx)
            self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
            self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
            self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
            self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)
            self.F_out = np.insert(self.F_out, closest_greater_index, np.zeros(self.F_out.shape[1]), axis=0)
            self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_end_idx])


class Force:
    """
    Force class. This class forms the base for defining external forces in solvers.
    """

    def __init__(self,
                 update_rhs_at_non_linear_iteration_func: Optional[callable] = None,
                 update_rhs_at_time_step_func: Optional[callable] = None):
        """
        Initializes the Force class with optional custom load functions.

        Args:
            update_rhs_at_non_linear_iteration_func (Optional[callable]): Callback to update forces per non-linear iteration.
            update_rhs_at_time_step_func (Optional[callable]): Callback to update forces per time step.
        """
        self.update_rhs_at_non_linear_iteration_func = update_rhs_at_non_linear_iteration_func
        self.update_rhs_at_time_step_func = update_rhs_at_time_step_func

        self.F = None
        self.force_matrix = None

    def initialise_stage(self, F: np.ndarray):
        """
        Initializes a calculation stage and derives per-step load functions.

        Args:
            F (np.ndarray): External force matrix or vector.
        """
        # if F is a matrix, initialise force_matrix
        if F.ndim == 2:
            self.force_matrix = F
        else:
            self.F = F

        # define load function, if none is given
        if self.update_rhs_at_time_step_func is None:
            def load_func(t: int, **kwargs):
                """
                Gets Force at time t from Force matrix

                Args:
                    t (int): Time index.
                    **kwargs: Additional keyword arguments forwarded to the custom load function.
                              This is required for self.update_rhs_at_time_step_func
                """
                if self.force_matrix is not None:
                    return self.force_matrix[:, t]
                else:
                    return self.F

            self.update_rhs_at_time_step_func = load_func

    def update_rhs_at_time_step(self, t: int, **kwargs):
        """
        Updates the force vector for a specific time step.

        Args:
            t (int): Time index.
            **kwargs: Additional keyword arguments forwarded to the custom load function.
        """
        self.F = self.update_rhs_at_time_step_func(t, **kwargs)

        # convert sparse matrix to a 1d vector
        if issparse(self.F):
            self.F = self.F.toarray()[:, 0]

    def update_rhs_at_non_linear_iteration(self, t: int, **kwargs):
        """
        Updates the force vector during non-linear iterations when a custom function is provided.

        Args:
            t (int): Time index.
            **kwargs: Additional keyword arguments forwarded to the custom load function.
        """
        # if a custom function is provided to update force at non linear iteration, update the force
        if self.update_rhs_at_non_linear_iteration_func is not None:
            self.F = self.update_rhs_at_non_linear_iteration_func(t, **kwargs)

        # convert sparse matrix to a 1d vector
        if issparse(self.F):
            self.F = self.F.toarray()[:, 0]

    def update_force(self,
                     u: npt.NDArray[np.float64],
                     F_previous: npt.NDArray[np.float64],
                     t: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes force increment and total force at time index t.

        Args:
            u (npt.NDArray[np.float64]): Displacement vector at time t.
            F_previous (npt.NDArray[np.float64]): Force vector from the previous time step.
            t (int): Current time-step index.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Force increment and total force.
        """
        # calculates force with custom load function
        self.update_rhs_at_non_linear_iteration(t, u=u)

        force = self.F

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)

        return d_force, F_total

    @staticmethod
    def validate_input(t_start_idx: int,
                       t_end_idx: int,
                       time: npt.NDArray[np.float64],
                       force_matrix: Optional[npt.NDArray[np.float64]]):
        """
        Validates force shape and uniform time steps in the current stage.

        Args:
            t_start_idx (int): First time index of the current stage.
            t_end_idx (int): Last time index of the current stage.
            time (npt.NDArray[np.float64]): Solver time discretisation.
            force_matrix (Optional[npt.NDArray[np.float64]]): External force matrix.
        """
        #
        # validate shape external force vector
        if force_matrix is not None:
            if len(time) != np.shape(force_matrix)[1]:
                raise ValueError("Solver time is not equal to force vector time")

        # validate time step size
        diff = np.diff(time[t_start_idx:t_end_idx])
        if diff.size > 0:
            if not np.all(np.isclose(diff, diff[0])):
                raise ValueError("Time steps differ in current stage")

def calculate_initial_acceleration(m_global: Matrix,
                                   c_global: Matrix,
                                   k_global: Matrix,
                                   force_ini: npt.NDArray[np.float64],
                                   u: npt.NDArray[np.float64],
                                   v: npt.NDArray[np.float64],
                                   linear_solver: LinearSolversABC,
                                   preconditioner: Optional[PreconditionerABC]) -> npt.NDArray[np.float64]:
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
