from scipy import sparse
from typing import Final

from solvers.utils import LumpingMethod
from solvers.linear_equations_solvers import (SparseDirectSolver, SparseDirectSolverLU, DenseDirectSolver,
    CGSolver, GMRESSolver, BICSTABSolver)
from solvers.preconditioners import JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner
from solvers.newmark_solver import NewmarkExplicit, NewmarkImplicitForce
from solvers.central_difference_solver import CentralDifferenceSolver
from solvers.bathe_solver import BatheSolver
from solvers.HHT_solver import HHTImplicitForce, HHTExplicit
from solvers.zhai_solver import ZhaiSolver


def has_cupy():
    try:
        import cupy
        return True
    except ImportError:
        return False


def set_matrices_as_sparse(M, K, C, F):
    M = sparse.csc_matrix(M)
    K = sparse.csc_matrix(K)
    C = sparse.csc_matrix(C)
    F = sparse.csc_matrix(F)

    return M, K, C, F


def set_matrices_as_np_array(M, K, C, F):
    if sparse.issparse(M):
        M = M.toarray()
    if sparse.issparse(K):
        K = K.toarray()
    if sparse.issparse(C):
        C = C.toarray()
    if sparse.issparse(M):
        F = F.toarray()

    return M, K, C, F


# variables for parametrization of the tests
ALL_LINEAR_SOLVERS: Final = (SparseDirectSolver, SparseDirectSolverLU, CGSolver, GMRESSolver, BICSTABSolver)
ALL_PRECONDITIONERS: Final = (None, JacobiPreconditioner, SSORPreconditioner, ILUPreconditioner)
ALL_LUMPING_METHODS: Final = (LumpingMethod.NONE, LumpingMethod.RowSum)
ALL_DENSE_LINEAR_SOLVERS: Final = (SparseDirectSolver, CGSolver, GMRESSolver, BICSTABSolver)
ALL_DYNAMIC_SOLVERS: Final = [NewmarkExplicit, NewmarkImplicitForce, CentralDifferenceSolver, BatheSolver, HHTImplicitForce, HHTExplicit, ZhaiSolver]