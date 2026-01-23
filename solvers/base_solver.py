from typing import Union, Optional, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import issparse, csc_matrix


# Define a custom type alias for readability
Matrix: TypeAlias = Union[npt.NDArray[np.float64], sp.spmatrix]


class State:
    """
    State class. This class forms the base for each solver.
    """
    def __init__(self):
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
        self.output_interval = 1
        self.F_out = None
        self.output_time = None
        self.output_time_indices = None
        self.number_equations = None


    def initialise(self, number_equations: int, time: npt.NDArray[np.float64]):
        """
        Initialises displacement and velocity vectors
        Initialises output time interval and output matrices

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

        match = np.where(self.output_time_indices == t_index)[0]
        if match.size == 0:
            return

        self.u[match[0]] = u
        self.v[match[0]] = v
        self.a[match[0]] = a
        self.f[match[0]] = f
        self.F_out[match[0]] = F

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
