from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np

# import numpy.typing as npt
# import scipy.sparse as sp

from scipy.sparse import issparse, csc_matrix
# from numpy.linalg import inv
# from scipy.sparse.linalg import LinearOperator
# from scipy.sparse.linalg import splu, spsolve_triangular, spilu, bicgstab, cg, spsolve, gmres
# from scipy.sparse import tril, triu
# # from solvers.utils import PreConditioner, Solvers

import numpy.typing as npt
import scipy.sparse as sp
from typing import Union, TypeAlias

# Define a custom type alias for readability
Matrix: TypeAlias = Union[npt.NDArray[np.float64], sp.spmatrix]


class State:
    """
    State class. This class forms the base for each solver.
    """
    def __init__(self):
        """
        Initializes the State class with default attributes.

        :Attributes:
            - :self.u0: initial displacement vector
            - :self.v0: initial velocity vector
            - :self.u: displacement matrix with size [ndof, number of time steps / output_interval]
            - :self.v: velocity matrix with size [ndof, number of time steps / output_interval]
            - :self.a: acceleration matrix with size [ndof, number of time steps / output_interval]
            - :self.f: nodal force matrix with size [ndof, number of time steps / output_interval]
            - :self.time: time discretisation
            - :self.output_interval: number of time steps interval in which output results are stored
            - :self.F_out: output external forces stored at self.output_interval
            - :self.output_time: output time discretisation stored at self.output_interval
            - :self.output_time_indices: time indices on which results are stored
            - :self.number_equations: number of equations to be solved
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


    def initialise(self, number_equations, time):
        """
        Initialises the solver before the calculation starts. Initialises displacement and velocity vectors; initialises
        output time interval and output matrices

        :param number_equations: number of equations to be solved
        :param time: time discretisation
        :return:
        """
        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

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

        self.number_equations = number_equations

    def check_for_sparse(self, M: Matrix, C: Matrix, K: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        """
        Checks if one of the input matrices is a sparse matrix.
        If so, convert all input matrices to csc sparse matrices

        :param M: mass matrix
        :param C: damping matrix
        :param K: stiffness matrix
        :return: Updated matrices M, C, K in csc format if any were sparse
        """
        # check if sparse calculation should be performed
        if issparse(M) or issparse(C) or issparse(K):
            M = csc_matrix(M)
            C = csc_matrix(C)
            K = csc_matrix(K)

        return M, C, K

    def update_output_arrays(self, t_start_idx: int, t_end_idx: int) -> None:
        """
        Updates output arrays.
        If either the t_start_idx or t_end_idx is missing in the output indices array, these indices are added.

        :param t_start_idx: start time index
        :param t_end_idx: end time index
        """

        # add start time index if required
        if t_start_idx not in self.output_time_indices:
            closest_greater_index = np.where(
                self.output_time_indices[self.output_time_indices > t_start_idx].min() == self.output_time_indices)[0]
            self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_start_idx)
            self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
            self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
            self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
            self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)

            self.F_out = np.insert(self.F_out, closest_greater_index, np.zeros(self.F_out.shape[1]), axis=0)

            self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_start_idx])

        # add end time index if required
        if t_end_idx not in self.output_time_indices:
            closest_greater_index = np.where(
                self.output_time_indices[self.output_time_indices > t_end_idx].min() == self.output_time_indices)[0]

            self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_end_idx)
            self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
            self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
            self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
            self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)

            self.F_out = np.insert(self.F_out, closest_greater_index, np.zeros(self.F_out.shape[1]), axis=0)

            self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_end_idx])

    def validate_input(self, t_start_idx: int, t_end_idx: int, force_matrix) -> None:
        """
        Validates solver input at current stage.
        It is checked if the external force vector shape corresponds with the time discretisation.
        Furthermore, it is checked if all time steps in the current stage are equal.

        :param t_start_idx: first time index of current stage
        :param t_end_idx:   last time index of current stage
        """
        #
        # validate shape external force vector
        if force_matrix is not None:
            if len(self.time) != np.shape(force_matrix)[1]:
                raise ValueError("Solver time is not equal to force vector time")

        # validate time step size
        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if diff.size > 0:
            if not np.all(np.isclose(diff, diff[0])):
                raise ValueError("Time steps differ in current stage")


class Force:
    """
    Force class. This class forms the base for defining external forces in solvers.
    """

    def __init__(self,
                 update_rhs_at_non_linear_iteration_func: Optional[callable] = None,
                 update_rhs_at_time_step_func: Optional[callable] = None) -> None:
        """
        Initializes the Force class with optional custom load functions.

        Parameters
        ----------
        update_rhs_at_non_linear_iteration_func : callable, optional
            Custom function to update the external force at each non-linear iteration.
        update_rhs_at_time_step_func : callable, optional
            Custom function to update the external force at each time step.
        """
        self.update_rhs_at_non_linear_iteration_func = update_rhs_at_non_linear_iteration_func
        self.update_rhs_at_time_step_func = update_rhs_at_time_step_func

        self.F = None
        self.force_matrix = None

    def initialise_stage(self, F: np.ndarray):
        """
        Initialises a calculation stage.
        It is checked if the external force is in matrix form or vector form.
        If the external force vector is in matrix form, a load function is generated which retrieves
        the force vector per time step from the force matrix.

        Parameters
        ----------
        F : np.ndarray
            External force matrix or vector
        """

        # if F is a matrix, initialise force_matrix
        if F.ndim == 2:
            self.force_matrix = F
        else:
            self.F = F

        # define load function, if none is given
        if self.update_rhs_at_time_step_func is None:
            def load_func(t: int, **kwargs) -> np.ndarray:
                """
                Gets Force at time t from Force matrix

                Parameters
                ----------
                t : int
                    time index
                kwargs : dict
                    key word arguments, this is required for self.update_rhs_at_time_step_func

                Returns
                -------
                np.ndarray
                    Force vector at time index t
                """
                if self.force_matrix is not None:
                    return self.force_matrix[:, t]
                else:
                    return self.F

            self.update_rhs_at_time_step_func = load_func

    def update_rhs_at_time_step(self, t: int, **kwargs):
        """
        Updates force vector at a time step

        Parameters
        ----------
        t : int
            time index
        kwargs : dict
            optional key word arguments
        """

        self.F = self.update_rhs_at_time_step_func(t, **kwargs)

        # convert sparse matrix to a 1d vector
        if issparse(self.F):
            self.F = self.F.toarray()[:, 0]

    def update_rhs_at_non_linear_iteration(self, t: int, **kwargs):
        """
        Updates force vector at a non-linear iteration, only if a custom function is provided

        Parameters
        ----------
        t : int
            time index
        kwargs : dict
            optional key word arguments
        """

        # if a custom function is provided to update force at non linear iteration, update the force
        if self.update_rhs_at_non_linear_iteration_func is not None:
            self.F = self.update_rhs_at_non_linear_iteration_func(t, **kwargs)

        # convert sparse matrix to a 1d vector
        if issparse(self.F):
            self.F = self.F.toarray()[:, 0]


    def update_force(self, u: npt.NDArray[np.float64], F_previous: npt.NDArray[np.float64], t: int):
        """
        Updates the external force vector at time t

        Parameters
        ----------
        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index

        :return: incremental force vector and total force vector
        """

        # calculates force with custom load function
        self.update_rhs_at_non_linear_iteration(t, u=u)

        force = self.F

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)

        return d_force, F_total


# class TimeException(Exception):
#     """
#     Raised when time steps in solver are not correct
#     """
#     pass


# class Solver:
#     """
#     Solver class. This class forms the base for each solver.

#     :Attributes:

#         - :self.u0:                     initial displacement vector
#         - :self.v0:                     initial velocity vector
#         - :self.u:                      displacement matrix with size [ndof, number of time steps / output_interval]
#         - :self.v:                      velocity matrix with size [ndof, number of time steps / output_interval]
#         - :self.a:                      acceleration matrix with size [ndof, number of time steps / output_interval]
#         - :self.f:                      nodal force matrix with size [ndof, number of time steps / output_interval]
#         - :self.time:                   time discretisation
#         - :self.update_rhs_at_non_linear_iteration_func:
#                                         optional custom load function to alter external force per non linear iteration
#         - :self.update_rhs_at_time_step_func:
#                                         optional custom load function to alter external force per time step
#         - :self.stiffness_func:         optional custom stiffness function to alter stiffness matrix during calculation
#         - :self.mass_func:              optional custom mass function to alter mass matrix during calculation
#         - :self.damping_func:           optional custom damping function to alter damping matrix during calculation
#         - :self.force_matrix:           external force vector
#         - :self.force_matrix:           external force matrix of size [ndof, number of time steps]
#         - :self.output_interval:        number of time steps interval in which output results are stored
#         - :self.F_out:                  output external forces stored at self.output_interval
#         - :self.output_time:            output time discretisation stored at self.output_interval
#         - :self.output_time_indices:    time indices on which results are stored
#         - :self.number_equations:       number of equations to be solved
#         - :self._is_sparse_calculation: bool which indicates if calculation should be performed with sparse solver
#     """

#     def __init__(self, preconditioner=PreConditioner.NONE, solver=Solvers.DIRECT):
#         # define initial conditions
#         self.u0 = []
#         self.v0 = []

#         # define variables
#         self.u = []
#         self.v = []
#         self.a = []
#         self.f = []
#         self.time = []

#         # load functions
#         self.update_rhs_at_non_linear_iteration_func = None
#         self.update_rhs_at_time_step_func = None
#         self.stiffness_func = None
#         self.mass_func = None
#         self.damping_func = None

#         self.F = None
#         self.force_matrix = None

#         self.output_interval = 1
#         self.F_out = []
#         self.output_time = []
#         self.output_time_indices = []

#         self.number_equations = None

#         self._is_sparse_calculation = None
#         self.cache_inv_K_till = None
#         self.preconditioner = preconditioner
#         self.solver = solver


#     def check_for_sparse(self, M, C, K):
#         """
#         Checks if one of the input matrices is a sparse matrix. If so, convert all input matrices to csc sparse matrices

#         :param M: mass matrix
#         :param C: damping matrix
#         :param K: stiffness matrix
#         :return:
#         """
#         # check if sparse calculation should be performed
#         if issparse(M) or issparse(C) or issparse(K):
#             self._is_sparse_calculation = True
#             Warning("Converting matrices to csc sparse matrices")

#             M = csc_matrix(M)
#             C = csc_matrix(C)
#             K = csc_matrix(K)
#         else:
#             self._is_sparse_calculation = False

#         return M, C, K

#     def initialise(self, number_equations, time):
#         """
#         Initialises the solver before the calculation starts. Initialises displacement and velocity vectors; initialises
#         output time interval and output matrices

#         :param number_equations: number of equations to be solved
#         :param time: time discretisation
#         :return:
#         """
#         self.u0 = np.zeros(number_equations)
#         self.v0 = np.zeros(number_equations)

#         self.time = np.array(time)

#         # find indices of time steps which should be stored based on output interval
#         self.output_time_indices = np.arange(0, len(self.time), self.output_interval)

#         # make sure last time step is included
#         if not np.isclose(self.output_time_indices[-1], len(self.time) - 1):
#             self.output_time_indices = np.append(self.output_time_indices, len(self.time) - 1)

#         # initialise result arrays
#         self.output_time = self.time[self.output_time_indices]
#         self.u = np.zeros((len(self.output_time_indices), number_equations))
#         self.v = np.zeros((len(self.output_time_indices), number_equations))
#         self.a = np.zeros((len(self.output_time_indices), number_equations))
#         self.f = np.zeros((len(self.output_time_indices), number_equations))

#         self.F_out = np.zeros((len(self.output_time_indices), number_equations))

#         self.number_equations = number_equations

#     def update(self, t_start_idx):
#         """
#         Updates the solver on a certain stage. Initial conditions are retrieved from previously calculated values for
#         displacements and velocities.

#         :param t_start_idx: start time index of current stage
#         :return:
#         """
#         output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]

#         self.u0 = self.u[output_time_idx, :]
#         self.v0 = self.v[output_time_idx, :]

#     def initialise_stage(self, F):
#         """
#         Initialises a calculation stage. It is checked if the external force is in matrix form or vector form. If the
#         external force vector is in matrix form, a load function is generated which retrieves the force vector per time
#         step from the force matrix.

#         :param F: external force matrix or vector
#         :return:
#         """

#         # if F is a matrix, initialise force_matrix
#         if F.ndim == 2:
#             self.force_matrix = F
#         else:
#             self.F = F

#         # define load function, if none is given
#         if self.update_rhs_at_time_step_func is None:
#             def load_func(t, **kwargs):
#                 """
#                 Gets Force at time t from Force matrix

#                 :param t: time index
#                 :param kwargs: key word arguments, this is required for self.update_rhs_at_time_step_func
#                 :return:
#                 """
#                 if self.force_matrix is not None:
#                     return self.force_matrix[:, t]
#                 else:
#                     return self.F

#             self.update_rhs_at_time_step_func = load_func

#     def update_rhs_at_time_step(self, t, **kwargs):
#         """
#         Updates force vector at a time step

#         :param t: time index
#         :param kwargs: optional key word arguments
#         :return:
#         """

#         self.F = self.update_rhs_at_time_step_func(t, **kwargs)

#         # convert sparse matrix to a 1d vector
#         if issparse(self.F):
#             self.F = self.F.toarray()[:, 0]

#     def update_rhs_at_non_linear_iteration(self, t, **kwargs):
#         """
#         Updates force vector at a non-linear iteration, only if a custom function is provided

#         :param t: time index
#         :param kwargs: optional key word arguments
#         :return:
#         """

#         # if a custom function is provided to update force at non linear iteration, update the force
#         if self.update_rhs_at_non_linear_iteration_func is not None:
#             self.F = self.update_rhs_at_non_linear_iteration_func(t, **kwargs)

#         # convert sparse matrix to a 1d vector
#         if issparse(self.F):
#             self.F = self.F.toarray()[:, 0]

#     def update_output_arrays(self, t_start_idx, t_end_idx):
#         """
#         Updates output arrays. If either the t_start_idx or t_end_idx is missing in the output indices array, these
#         indices are added.

#         :param t_start_idx: start time index
#         :param t_end_idx: end time index
#         :return:
#         """

#         # add start time index if required
#         if t_start_idx not in self.output_time_indices:
#             closest_greater_index = np.where(
#                 self.output_time_indices[self.output_time_indices > t_start_idx].min() == self.output_time_indices)[0]
#             self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_start_idx)
#             self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
#             self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
#             self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
#             self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)

#             self.F_out = np.insert(self.F_out, closest_greater_index, np.zeros(self.F_out.shape[1]), axis=0)

#             self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_start_idx])

#         # add end time index if required
#         if t_end_idx not in self.output_time_indices:
#             closest_greater_index = np.where(
#                 self.output_time_indices[self.output_time_indices > t_end_idx].min() == self.output_time_indices)[0]

#             self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_end_idx)
#             self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
#             self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
#             self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
#             self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)

#             self.F_out = np.insert(self.F_out, closest_greater_index, np.zeros(self.F_out.shape[1]), axis=0)

#             self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_end_idx])

#     def finalise(self):
#         """
#         Finalises the solver. Displacements, velocities, accelerations and time are stored at a certain interval.
#         :return:
#         """
#         pass

#     def validate_input(self, t_start_idx, t_end_idx):
#         """
#         Validates solver input at current stage. It is checked if the external force vector shape corresponds with the
#         time discretisation. Furthermore, it is checked if all time steps in the current stage are equal.

#         :param t_start_idx: first time index of current stage
#         :param t_end_idx:   last time index of current stage
#         :return:
#         """
#         #
#         # validate shape external force vector
#         if self.force_matrix is not None:
#             if len(self.time) != np.shape(self.force_matrix)[1]:
#                 logging.error("Solver error: Solver time is not equal to force vector time")
#                 raise TimeException("Solver time is not equal to force vector time")

#         # validate time step size
#         diff = np.diff(self.time[t_start_idx:t_end_idx])
#         if diff.size > 0:
#             if not np.all(np.isclose(diff, diff[0])):
#                 logging.error("Solver error: Time steps differ in current stage")
#                 raise TimeException("Time steps differ in current stage")

#     def solve_linear_system(self, k_till, delta_force, cache=True):
#         """
#         Solves a linear system of equations of the form K *  ∆u = ∆F, where K is the equivalent stiffness matrix,
#         ∆u is the unknown incremental vector to be solved for, and ∆F is the incremental external force vector.

#         :param k_till: equivalent stiffness matrix
#         :param delta_force: change in external incremental force vector
#         :param cache: bool to indicate if the inverse of k_till should be cached for future use
#         :return: solution incremental vector ∆u, info flag
#         """

#         # pre-conditioners and iterative solvers only work with sparse matrices
#         if self._is_sparse_calculation:
#             if self.preconditioner.value is None:
#                 if self.cache_inv_K_till is None:
#                     self.cache_inv_K_till = splu(k_till)
#                 du = self.cache_inv_K_till.solve(delta_force)
#                 info = 0
#             else:
#                 if self.solver == Solvers.DIRECT:
#                     solver = self.solver.apply()
#                     du = solver(k_till, delta_force)
#                     info = 0
#                 else:
#                     M_op = self.preconditioner.apply()
#                     solver = self.solver.apply()
#                     du, info = solver(k_till, delta_force, M=M_op, rtol=1e-12, maxiter=10_000)
#         else:
#             if self.cache_inv_K_till is None:
#                 self.cache_inv_K_till = inv(k_till)
#             du = self.cache_inv_K_till.dot(delta_force)
#             info = 0

#         if cache==False:
#             self.cache_inv_K_till = None
#         return du, info