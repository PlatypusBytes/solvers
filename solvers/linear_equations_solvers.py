import warnings
from typing import Union, Optional
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import bicgstab, cg, spsolve, gmres

from solvers.preconditioners import PreconditionerABC


class LinearSolversABC(ABC):
    """
    Abstract base class for solvers
    """
    @abstractmethod
    def solve(self,
              A: Union[npt.NDArray[np.float64], sp.spmatrix],
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve Ax = b

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).
        cache : bool, optional
            Whether to cache factorization for repeated timesteps (only for direct solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        raise NotImplementedError("Subclasses should implement this!")


class DenseDirectSolver(LinearSolversABC):
    """
    Direct dense solver using LU factorization (scipy.linalg.lu_factor).
    Caches the LU factorization to speed up repeated solves with the same A.
    """
    def __init__(self):
        """
        Initializes the direct dense solver with optional caching.
        """
        self._lu = None
        self._piv = None
        self._A_id = None

    def solve(self,
              A: npt.NDArray[np.float64],
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve A x = b using direct inversion.

        Parameters
        ----------
        A : np.ndarray
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """

        if M is not None:
            warnings.warn("Preconditioner is ignored in DenseDirectSolver")

        # identity-based cache: cache LU factorization if A is the same as last time
        if get_numpy_fingerprint(A) != self._A_id:
            self._lu, self._piv = lu_factor(A)
            self._A_id = get_numpy_fingerprint(A)
        x = lu_solve((self._lu, self._piv), b)
        return x

    def __getstate__(self) -> dict:
        """
        Prepare the state for pickling.
        This is needed for when doing deepcopy or pickling the solver.

        Returns:
            dict: The state dictionary without unpicklable entries.
        """
        state = self.__dict__.copy()
        state["_A_id"] = None
        state["_lu"] = None
        state["_piv"] = None
        return state

    def __setstate__(self, state: dict):
        """
        Restore the state from pickling.
        This is needed for when doing deepcopy or unpickling the solver.

        Args:
            state (dict): The state dictionary.
        """
        self.__dict__.update(state)
        self._A_id = None
        self._lu = None
        self._piv = None

class SparseDirectSolverLU(LinearSolversABC):
    """
    Sparse Direct Solver using scipy.sparse.linalg.spsolve
    """
    def __init__(self):
        """
        Initializes the Sparse Direct solver.
        """
        self._inverse_A = None
        self._A_id = None

    def solve(self,
              A: sp.spmatrix,
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve Ax = b using using direct inversion.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        if M is not None:
            warnings.warn("Preconditioner is ignored in SparseDirectSolverLU")

        # identity-based cache: cache inv A if A is the same as last time
        if get_sparse_fingerprint(A) != self._A_id:
            self._inverse_A = sp.linalg.splu(A)
            self._A_id = get_sparse_fingerprint(A)

        x = self._inverse_A.solve(b)
        return x

    def __getstate__(self) -> dict:
        """
        Prepare the state for pickling.
        This is needed for when doing deepcopy or pickling the solver.

        Returns:
            dict: The state dictionary without unpicklable entries.
        """
        state = self.__dict__.copy()
        state["_inverse_A"] = None
        state["_A_id"] = None
        return state

    def __setstate__(self, state: dict):
        """
        Restore the state from pickling.
        This is needed for when doing deepcopy or unpickling the solver.

        Args:
            state (dict): The state dictionary.
        """
        self.__dict__.update(state)
        self._inverse_A = None
        self._A_id = None




class SparseDirectSolver(LinearSolversABC):
    """
    Sparse Direct Solver using scipy.sparse.linalg.spsolve
    """
    def __init__(self):
        """
        Initializes the Sparse Direct solver.
        """
        pass

    def solve(self,
              A: sp.spmatrix,
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve Ax = b using using direct inversion.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        if M is not None:
            warnings.warn("Preconditioner is ignored in SparseDirectSolver")
        x = spsolve(A, b)
        return x


class CGSolver(LinearSolversABC):
    """
    Conjugate Gradient Solver
    """
    def __init__(self, rtol: float = 1e-12, maxiter: int =10_000):
        """
        Initializes the CG solver with relative tolerance and maximum iterations.
        """
        self.rtol = rtol
        self.maxiter = maxiter

    def solve(self,
              A: sp.spmatrix,
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve Ax = b using the Conjugate Gradient method.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        x, info = cg(A, b, M=M, rtol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return x


class GMRESSolver(LinearSolversABC):
    """
    Generalized Minimal Residual (GMRES) Solver
    """
    def __init__(self, rtol: float = 1e-12, maxiter: int =10_000):
        """
        Initializes the GMRES solver with relative tolerance and maximum iterations.
        """
        self.rtol = rtol
        self.maxiter = maxiter

    def solve(self,
              A: sp.spmatrix,
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve Ax = b using the Generalized Minimal Residual (GMRES) method.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        x, info = gmres(A, b, M=M, rtol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
        return x

class BICSTABSolver(LinearSolversABC):
    """
    Biconjugate Gradient Stabilized (BiCGSTAB) Solver
    """
    def __init__(self, rtol: float = 1e-12, maxiter: int =10_000):
        """
        Initializes the BiCGSTAB solver with relative tolerance and maximum iterations.
        """
        self.rtol = rtol
        self.maxiter = maxiter

    def solve(self,
              A: sp.spmatrix,
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              ) -> npt.NDArray[np.float64]:
        """
        Solve Ax = b using the Biconjugate Gradient Stabilized (BiCGSTAB) method.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix.
        b : np.ndarray
            The right-hand side vector.
        M : LinearOperator, optional
            Preconditioner (only for iterative solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        x, info = bicgstab(A, b, M=M, rtol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"BiCGSTAB did not converge (info={info})")
        return x


def get_sparse_fingerprint(A: sp.spmatrix) -> tuple:
    """
    Get a fingerprint of a sparse matrix A to identify it uniquely.

    Args:
        A (sp.spmatrix): The sparse matrix.
    Returns:
        tuple: A tuple containing the id, shape, number of non-zeros, and memory pointers of data and indices.
    """
    return (
        id(A),
        A.shape,
        A.nnz,
        A.data.ctypes.data,
        A.indices.ctypes.data
    )

def get_numpy_fingerprint(A: npt.NDArray[np.float64]) -> tuple:
    """
    Get a fingerprint of a numpy array A to identify it uniquely.

    Args:
        A (np.ndarray): The numpy array.
    Returns:
        tuple: A tuple containing the id, shape, dtype, strides, and memory address of the first element.
    """
    return (
        id(A),
        A.shape,
        A.dtype,
        A.strides,
        A.ctypes.data
    )