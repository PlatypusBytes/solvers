from typing import Optional
import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse
from tqdm import tqdm

from solvers.linear_equations_solvers import LinearSolversABC, SparseDirectSolverLU
from solvers.preconditioners import PreconditionerABC
from solvers.base_solver import BaseSolverABC, Force, State, Matrix, TimeIntegrationType


class StaticSolver(BaseSolverABC):
    """
    Static Solver class.
    """
    def __init__(self,
                 force: Optional[Force] = None,
                 state: Optional[State] = None,
                 linear_solver: Optional[LinearSolversABC] = None,
                 preconditioner: Optional[PreconditionerABC] = None):
        """
        Constructor of the Static Solver class.

        Args:
            force (Optional[Force]): Force definitions (default: None)
            state (Optional[State]): Solver state (default: None)
            linear_solver (Optional[LinearSolversABC]): Linear solver to be used (default: None)
            preconditioner (Optional[PreconditionerABC]): Preconditioner to be used (default: None)
        """
        self.linear_solver = linear_solver if linear_solver is not None else SparseDirectSolverLU()
        self.preconditioner = preconditioner
        self.force = force if force is not None else Force()
        self.state = state if state is not None else State()

    @property
    def type(self) -> TimeIntegrationType:
        """
        Time integration type of the solver.
        """
        return TimeIntegrationType.STATIC

    def initialise(self, number_eq: int, time: npt.NDArray[np.float64]):
        """
        Initialise the solver state.

        Args:
            number_eq (int): Number of equations
            time (npt.NDArray[np.float64]): Time array
        """
        self.state.initialise(number_eq, time)

    def calculate(self, K: Matrix, F: Matrix, t_start_idx: int, t_end_idx: int, F_ini: Optional[Matrix] = None):
        """
        Perform calculation with the Static solver incremental formulation.

        Args:
            K (Matrix): Stiffness matrix
            F (Matrix): External force matrix
            t_start_idx (int): Start time index
            t_end_idx (int): End time index
            F_ini (Optional[Matrix], optional): Initial external force matrix. Defaults to None.
        """

        # initialize force for the stage
        self.force.initialise_stage(F)
        # validate force input
        self.force.validate_input(t_start_idx, t_end_idx, self.state.time, self.force.force_matrix)

        # update output arrays for the stage
        self.state.update_output_arrays(t_start_idx, t_end_idx)

        # initial conditions u
        u = self.state.u0

        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=u)

        output_time_idx = np.where(self.state.output_time_indices == t_start_idx)[0][0]
        t2 = output_time_idx + 1

        # add to results initial conditions
        self.state.store_step(output_time_idx, u, None, None, None, self.force.F)

        # define progress bar
        pbar = tqdm(total=(t_end_idx - t_start_idx), unit_scale=True, unit_divisor=1000, unit="steps")

        self.force.update_rhs_at_time_step(t_start_idx)
        self.force.update_rhs_at_non_linear_iteration(t_start_idx, u=u)

        # set initial incremental external force
        if F_ini is None:
            F_ini = np.zeros_like(self.force.F)
        else:
            if issparse(F_ini):
                F_ini = F_ini.toarray()[:, 0]

        d_force_ini = self.force.F - F_ini
        F_prev = np.copy(self.force.F)

        # Build preconditioner if provided
        if self.preconditioner is not None:
            pre_c = self.preconditioner.build(K)
        else:
            pre_c = None

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            self.force.update_rhs_at_time_step(t)
            self.force.update_rhs_at_non_linear_iteration(t)

            # update external force
            d_force = d_force_ini + self.force.F - F_prev

            # solve
            uu = self.linear_solver.solve(K, d_force, M=pre_c)

            # update displacement
            u = u + uu

            # add to results
            # add to results
            if t == self.state.output_time_indices[t2]:
                self.state.store_step(t2, u, None, None, K @ u, self.force.F)
                t2 += 1

            d_force_ini = 0
            F_prev = np.copy(self.force.F)

        # close the progress bar
        pbar.close()
