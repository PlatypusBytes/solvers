import numpy as np
from scipy.sparse import issparse, csc_matrix
from pypardiso import spsolve as pypardiso_solver
from scipy.sparse.linalg import spsolve as scipy_solver

import os
import pickle
from tqdm import tqdm

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

        - :self.u0:                 initial displacement vector
        - :self.v0:                 initial velocity vector
        - :self.u:                  full displacement matrix [ndof, number of time steps]
        - :self.v:                  full velocity matrix [ndof, number of time steps]
        - :self.a:                  full acceleration matrix [ndof, number of time steps]
        - :self.f:                  full force matrix [ndof, number of time steps]
        - :self.time:               time discretisation
        - :self.load_func:          optional custom load function to alter external force during calculation
        - :self.stiffness_func:     optional custom stiffness function to alter stiffness matrix during calculation
        - :self.mass_func:          optional custom mass function to alter mass matrix during calculation
        - :self.damping_func:       optional custom damping function to alter damping matrix during calculation
        - :self.output_interval:    number of time steps interval in which output results are stored
        - :self.u_out:              output displacement stored at self.output_interval
        - :self.v_out:              output velocities stored at self.output_interval
        - :self.a_out:              output accelerations stored at self.output_interval
        - :self.time_out:           output time discretisation stored at self.output_interval
        - :self.number_equations:   number of equations to be solved
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

        # load function
        self.update_non_linear_iteration_rhs_func = None
        self.update_time_step_func = None
        self.stiffness_func = None
        self.mass_func = None
        self.damping_func = None

        self.force_matrix = None

        self.absorbing_boundary = None

        self.output_interval = 1
        self.u_out = []
        self.v_out = []
        self.a_out = []
        self.time_out = []

        self.output_time = []
        self.output_time_indices = []

        self.number_equations = None

        self._is_sparse_calculation = None
        self.sparse_solver = pypardiso_solver # pypardiso_solver or scipy_solver

    def check_for_sparse(self, M, C, K):
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
        Initialises the solver before the calculation starts

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

        if F.ndim == 2:
            self.force_matrix = F
        else:
            self.F = F

        if self.update_time_step_func is None:
            # define load function, if none is given
            def load_func(t, **kwargs):
                if self.force_matrix is not None:
                    return self.force_matrix[:, t]
                else:
                    return self.F

            self.update_time_step_func = load_func

    def update_time_step_rhs(self,t, **kwargs):


        self.F = self.update_time_step_func(t, **kwargs)
        # if self.update_time_step_func is None:
        #     # define load function, if none is given
        #     def load_func(t,**kwargs):
        #         if self.force_matrix is not None:
        #             return self.force_matrix[:,t]
        #         else:
        #             return self.F
        #     self.update_time_step_func = load_func
       # self.update_non_linear_iteration_rhs(t, **kwargs)

    def update_non_linear_iteration_rhs(self, t, **kwargs):

        if self.update_non_linear_iteration_rhs_func is not None:
            self.F = self.update_non_linear_iteration_rhs_func(t, **kwargs)

        if issparse(self.F):
            self.F = self.F.toarray()[:, 0]



    def update_output_arrays(self,t_start_idx, t_end_idx):
        """
        Updates output arrays. If either the t_start_idx or t_end_idx is missing in the output indices array, these indices
        are added.
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

            self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_start_idx])

        # add end time index if required
        if t_end_idx not in self.output_time_indices:
            closest_greater_index = np.where(self.output_time_indices[self.output_time_indices >t_end_idx].min() == self.output_time_indices)[0]
            self.output_time_indices = np.insert(self.output_time_indices, closest_greater_index, t_end_idx)
            self.u = np.insert(self.u, closest_greater_index, np.zeros(self.u.shape[1]), axis=0)
            self.v = np.insert(self.v, closest_greater_index, np.zeros(self.v.shape[1]), axis=0)
            self.a = np.insert(self.a, closest_greater_index, np.zeros(self.a.shape[1]), axis=0)
            self.f = np.insert(self.f, closest_greater_index, np.zeros(self.f.shape[1]), axis=0)

            self.output_time = np.insert(self.output_time, closest_greater_index, self.time[t_end_idx])

    def finalise(self):
        """
        Finalises the solver. Displacements, velocities, accelerations and time are stored at a certain interval.
        :return:
        """
        self.time_out = self.time[self.output_time_indices]

    def validate_input(self, F, t_start_idx, t_end_idx):
        """
        Validates solver input at current stage. It is checked if the external force vector shape corresponds with the
        time discretisation. Furthermore, it is checked if all time steps in the current stage are equal.

        :param F:           External force vector.
        :param t_start_idx: first time index of current stage
        :param t_end_idx:   last time index of current stage
        :return:
        """
        #
        # # validate shape external force vector
        # if len(self.time) != np.shape(F)[1]:
        #     logging.error("Solver error: Solver time is not equal to force vector time")
        #     raise TimeException("Solver time is not equal to force vector time")

        # validate time step size
        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if diff.size > 0:
            if not np.all(np.isclose(diff, diff[0])):
                logging.error("Solver error: Time steps differ in current stage")
                raise TimeException("Time steps differ in current stage")

if __name__ == '__main__':
    solver = Solver()

    solver.force_matrix = np.ones((2,3))

    solver.update_time_step(1)