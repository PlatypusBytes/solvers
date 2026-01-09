import warnings
from typing import Union, Optional
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, cg, spsolve, gmres

from solvers.preconditioners import PreconditionerABC


class SolversABC(ABC):
    """
    Abstract base class for solvers
    """
    @abstractmethod
    def solve(self,
              A: Union[npt.NDArray[np.float64], sp.spmatrix],
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              cache: bool = False) -> npt.NDArray[np.float64]:
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


class DenseDirectSolver(SolversABC):
    """
    Direct dense solver using numpy.linalg.inv
    """
    def __init__(self):
        """
        Initializes the direct dense solver with optional caching.
        """
        self.__invA_cached = None

    def solve(self,
              A: npt.NDArray[np.float64],
              b: npt.NDArray[np.float64],
              M: Optional[PreconditionerABC] = None,
              cache: bool = False) -> npt.NDArray[np.float64]:
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
        cache : bool, optional
            Whether to cache factorization for repeated timesteps (only for direct solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """

        if M is not None:
            warnings.warn("Preconditioner is ignored in DenseDirectSolver")

        if self.__invA_cached is None:
            self.__invA_cached = np.linalg.inv(A)
        x = self.__invA_cached.dot(b)

        if cache == False:
            self.__invA_cached = None
        return x


class SparseDirectSolver(SolversABC):
    """
    Conjugate Gradient Solver
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
              cache: bool = False) -> npt.NDArray[np.float64]:
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
        cache : bool, optional
            Whether to cache factorization for repeated timesteps (only for direct solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        if M is not None:
            warnings.warn("Preconditioner is ignored in SparseDirectSolver")
        x = spsolve(A, b)
        return x


class CGSolver(SolversABC):
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
              cache: bool = False) -> npt.NDArray[np.float64]:
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
        cache : bool, optional
            Whether to cache factorization for repeated timesteps (only for direct solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        x, info = cg(A, b, M=M, rtol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return x


class GMRESSolver(SolversABC):
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
              cache: bool = False) -> npt.NDArray[np.float64]:
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
        cache : bool, optional
            Whether to cache factorization for repeated timesteps (only for direct solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        x, info = gmres(A, b, M=M, rtol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
        return x

class BICSTABSolver(SolversABC):
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
              cache: bool = False) -> npt.NDArray[np.float64]:
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
        cache : bool, optional
            Whether to cache factorization for repeated timesteps (only for direct solvers).

        Returns
        -------
        x : np.ndarray
            The solution vector.
        """
        x, info = bicgstab(A, b, M=M, rtol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"BiCGSTAB did not converge (info={info})")
        return x
