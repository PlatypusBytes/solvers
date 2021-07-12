import numpy as np
from scipy.sparse import issparse, csc_matrix

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
        self.load_func = None
        self.stiffness_func = None
        self.mass_func = None
        self.damping_func = None

        self.absorbing_boundary = None

        self.output_interval = 10
        self.u_out = []
        self.v_out = []
        self.a_out = []
        self.time_out = []

        self.number_equations = None

        self._is_sparse_calculation = None

    def check_for_sparse(self, M, C, K):
        # check if sparse calculation should be performed
        if issparse(M) and issparse(C) and issparse(K):
            self._is_sparse_calculation = True
        elif issparse(M) or issparse(C) or issparse(K):
            self._is_sparse_calculation = True

            Warning("One but not all matrices is sparse, converting other matrices to csc sparse matrices")

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

        self.u = np.zeros((len(time), number_equations))
        self.v = np.zeros((len(time), number_equations))
        self.a = np.zeros((len(time), number_equations))
        self.f = np.zeros((len(time), number_equations))

        self.number_equations = number_equations

    def update(self, t_start_idx):
        """
        Updates the solver on a certain stage. Initial conditions are retrieved from previously calculated values for
        displacements and velocities.

        :param t_start_idx: start time index of current stage
        :return:
        """
        self.u0 = self.u[t_start_idx, :]
        self.v0 = self.v[t_start_idx, :]

    def finalise(self):
        """
        Finalises the solver. Displacements, velocities, accelerations and time are stored at a certain interval.
        :return:
        """
        self.u_out = self.u[0::self.output_interval,:]
        self.v_out = self.v[0::self.output_interval,:]
        self.a_out = self.a[0::self.output_interval,:]

        self.time_out = self.time[0::self.output_interval]

    def validate_input(self, F, t_start_idx, t_end_idx):
        """
        Validates solver input at current stage. It is checked if the external force vector shape corresponds with the
        time discretisation. Furthermore, it is checked if all time steps in the current stage are equal.

        :param F:           External force vector.
        :param t_start_idx: first time index of current stage
        :param t_end_idx:   last time index of current stage
        :return:
        """

        # validate shape external force vector
        if len(self.time) != np.shape(F)[1]:
            logging.error("Solver error: Solver time is not equal to force vector time")
            raise TimeException("Solver time is not equal to force vector time")

        # validate time step size
        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if diff.size > 0:
            if not np.all(np.isclose(diff, diff[0])):
                logging.error("Solver error: Time steps differ in current stage")
                raise TimeException("Time steps differ in current stage")