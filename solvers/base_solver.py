import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.sparse.linalg import spsolve

import logging


class TimeException(Exception):
    """
    Raised when time steps in solver are not correct
    """
    pass


class Solver:
    """
    Solver class. This class forms the base for each solver.

    :Attributes:

        - :self.u0:                     initial displacement vector
        - :self.v0:                     initial velocity vector
        - :self.u:                      displacement matrix with size [ndof, number of time steps / output_interval]
        - :self.v:                      velocity matrix with size [ndof, number of time steps / output_interval]
        - :self.a:                      acceleration matrix with size [ndof, number of time steps / output_interval]
        - :self.f:                      nodal force matrix with size [ndof, number of time steps / output_interval]
        - :self.time:                   time discretisation
        - :self.update_rhs_at_non_linear_iteration_func:
                                        optional custom load function to alter external force per non linear iteration
        - :self.update_rhs_at_time_step_func:
                                        optional custom load function to alter external force per time step
        - :self.stiffness_func:         optional custom stiffness function to alter stiffness matrix during calculation
        - :self.mass_func:              optional custom mass function to alter mass matrix during calculation
        - :self.damping_func:           optional custom damping function to alter damping matrix during calculation
        - :self.force_matrix:           external force vector
        - :self.force_matrix:           external force matrix of size [ndof, number of time steps]
        - :self.output_interval:        number of time steps interval in which output results are stored
        - :self.F_out:                  output external forces stored at self.output_interval
        - :self.output_time:            output time discretisation stored at self.output_interval
        - :self.output_time_indices:    time indices on which results are stored
        - :self.number_equations:       number of equations to be solved
        - :self._is_sparse_calculation: bool which indicates if calculation should be performed with sparse solver
    """

    def __init__(self):
        # define initial conditions
        self.u0 = []
        self.v0 = []

        # define variables
        self.u = []
        self.v = []
        self.a = []
        self.f = []
        self.time = []

        # load functions
        self.update_rhs_at_non_linear_iteration_func = None
        self.update_rhs_at_time_step_func = None
        self.stiffness_func = None
        self.mass_func = None
        self.damping_func = None

        self.F = None
        self.force_matrix = None

        self.output_interval = 1
        self.F_out = []
        self.output_time = []
        self.output_time_indices = []

        self.number_equations = None

        self._is_sparse_calculation = None
        self.sparse_solver = spsolve

    def check_for_sparse(self, M, C, K):
        """
        Checks if one of the input matrices is a sparse matrix. If so, convert all input matrices to csc sparse matrices

        :param M: mass matrix
        :param C: damping matrix
        :param K: stiffness matrix
        :return:
        """
        # check if sparse calculation should be performed
        if issparse(M) or issparse(C) or issparse(K):
            self._is_sparse_calculation = True
            Warning("Converting matrices to csc sparse matrices")

            M = csc_matrix(M)
            C = csc_matrix(C)
            K = csc_matrix(K)
        else:
            self._is_sparse_calculation = False

        return M, C, K

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
        self.output_time_indices = np.arange(0,len(self.time),self.output_interval)

        # make sure last time step is included
        if not np.isclose(self.output_time_indices[-1], len(self.time)-1):
            self.output_time_indices = np.append(self.output_time_indices, len(self.time)-1)

        # initialise result arrays
        self.output_time = self.time[self.output_time_indices]
        self.u = np.zeros((len(self.output_time_indices), number_equations))
        self.v = np.zeros((len(self.output_time_indices), number_equations))
        self.a = np.zeros((len(self.output_time_indices), number_equations))
        self.f = np.zeros((len(self.output_time_indices), number_equations))

        self.F_out = np.zeros((len(self.output_time_indices), number_equations))

        self.number_equations = number_equations

    def update(self, t_start_idx):
        """
        Updates the solver on a certain stage. Initial conditions are retrieved from previously calculated values for
        displacements and velocities.

        :param t_start_idx: start time index of current stage
        :return:
        """
        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]

        self.u0 = self.u[output_time_idx, :]
        self.v0 = self.v[output_time_idx, :]

    def initialise_stage(self, F):
        """
        Initialises a calculation stage. It is checked if the external force is in matrix form or vector form. If the
        external force vector is in matrix form, a load function is generated which retrieves the force vector per time
        step from the force matrix.

        :param F: external force matrix or vector
        :return:
        """

        # if F is a matrix, initialise force_matrix
        if F.ndim == 2:
            self.force_matrix = F
        else:
            self.F = F

        # define load function, if none is given
        if self.update_rhs_at_time_step_func is None:
            def load_func(t, **kwargs):
                """
                Gets Force at time t from Force matrix
                :param t: time index
                :param kwargs: optional key word arguments
                :return:
                """
                if self.force_matrix is not None:
                    return self.force_matrix[:, t]
                else:
                    return self.F

            self.update_rhs_at_time_step_func = load_func

    def update_rhs_at_time_step(self,t, **kwargs):
        """
        Updates force vector at a time step

        :param t: time index
        :param kwargs: optional key word arguments
        :return:
        """

        self.F = self.update_rhs_at_time_step_func(t, **kwargs)

    def update_rhs_at_non_linear_iteration(self, t, **kwargs):
        """
        Updates force vector at a non-linear iteration, only if a custom function is provided

        :param t: time index
        :param kwargs: optional key word arguments
        :return:
        """

        # if a custom function is provided to update force at non linear iteration, update the force
        if self.update_rhs_at_non_linear_iteration_func is not None:
            self.F = self.update_rhs_at_non_linear_iteration_func(t, **kwargs)

        # convert sparse matrix to a 1d vector
        if issparse(self.F):
            self.F = self.F.toarray()[:, 0]

    def update_output_arrays(self,t_start_idx, t_end_idx):
        """
        Updates output arrays. If either the t_start_idx or t_end_idx is missing in the output indices array, these
        indices are added.

        :param t_start_idx: start time index
        :param t_end_idx: end time index
        :return:
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

    def finalise(self):
        """
        Finalises the solver. Displacements, velocities, accelerations and time are stored at a certain interval.
        :return:
        """
        pass

    def validate_input(self, t_start_idx, t_end_idx):
        """
        Validates solver input at current stage. It is checked if the external force vector shape corresponds with the
        time discretisation. Furthermore, it is checked if all time steps in the current stage are equal.

        :param t_start_idx: first time index of current stage
        :param t_end_idx:   last time index of current stage
        :return:
        """
        #
        # validate shape external force vector
        if self.force_matrix is not None:
            if len(self.time) != np.shape(self.force_matrix)[1]:
                logging.error("Solver error: Solver time is not equal to force vector time")
                raise TimeException("Solver time is not equal to force vector time")

        # validate time step size
        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if diff.size > 0:
            if not np.all(np.isclose(diff, diff[0])):
                logging.error("Solver error: Time steps differ in current stage")
                raise TimeException("Time steps differ in current stage")