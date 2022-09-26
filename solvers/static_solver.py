from solvers.base_solver import Solver

import os
import pickle

import numpy as np
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import issparse, csc_matrix

from tqdm import tqdm
import logging


class StaticSolver(Solver):
    """
    Static Solver class. This class contains the static incremental solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    """

    def __init__(self):
        super(StaticSolver, self).__init__()

    def calculate(self, K, F, t_start_idx, t_end_idx, F_ini=None):
        """
        Static integration scheme.
        Incremental formulation.

        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
        """

        self.initialise_stage(F)

        # initial conditions u
        u = self.u0

        output_time_idx = np.where(self.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.u[output_time_idx, :] = u

        # validate input
        self.validate_input(t_start_idx, t_end_idx)

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        self.update_time_step_rhs(t_start_idx)
        self.update_non_linear_iteration_rhs(t_start_idx)

        # set initial incremental external force
        if F_ini is None:
            F_ini = np.zeros_like(self.F)
        # if issparse(F[:, 0]):
        #     F_ini = csc_matrix(F_ini).T
        d_force_ini= self.F - F_ini
        F_prev = np.copy(self.F)

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            self.update_time_step_rhs(t)
            self.update_non_linear_iteration_rhs(t)

            # update external force
            d_force = d_force_ini + self.F - F_prev

            # solve
            uu = spsolve(K, d_force)

            # update displacement
            u = u + uu

            # add to results
            if t == self.output_time_indices[t2]:
                self.u[t2, :] = u
                t2 += 1

            d_force_ini = 0
            F_prev = np.copy(self.F)

        # close the progress bar
        pbar.close()
