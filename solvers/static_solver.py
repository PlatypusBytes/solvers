from solvers.base_solver import Solver

import os
import pickle

import numpy as np
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import issparse

from tqdm import tqdm
import logging


class StaticSolver(Solver):
    """
    Static Solver class. This class contains the static incremental solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    """

    def __init__(self):
        super(StaticSolver, self).__init__()

    def calculate(self, K, F, t_start_idx, t_end_idx):
        """
        Static integration scheme.
        Incremental formulation.

        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
        """

        # initial conditions u
        u = self.u0
        # add to results initial conditions
        self.u[t_start_idx, :] = u
        # initial differential force
        if t_start_idx > 0:
            d_force = F[:, t_start_idx] - F[:, t_start_idx - 1]
        else:
            d_force = F[:, t_start_idx]

        # validate input
        self.validate_input(F, t_start_idx, t_end_idx)

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            # solve
            uu = spsolve(K, d_force)

            # update displacement
            u = u + uu

            # update external force
            d_force = F[:, t] - F[:, t - 1]

            # add to results
            self.u[t, :] = u

        # close the progress bar
        pbar.close()
