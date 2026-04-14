from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import tril, triu
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve_triangular, spilu
try:
    import cupyx.scipy.sparse as cps
except ImportError:
    cps = None

class PreconditionerABC(ABC):
    """
    Abstract base class for preconditioners
    """
    @abstractmethod
    def build(self, A: sp.spmatrix):
        """
        Build the preconditioner for sparse matrix A

        Args:
            A (sp.spmatrix): The system sparse matrix to precondition.
        """
        raise NotImplementedError("Subclasses should implement this!")

class JacobiPreconditioner(PreconditionerABC):
    """
    Jacobi preconditioner: P = diag(A)
    """
    def build(self, A: sp.spmatrix) -> LinearOperator:
        """
        Builds the Jacobi preconditioner for sparse matrix A

        Args:
            A (sp.spmatrix): The system sparse matrix to precondition.

        Returns:
            LinearOperator representing the preconditioner (approximation of A^{-1})
        """
        diag = A.diagonal()
        M_op = LinearOperator(shape=A.shape, matvec=lambda v: v / diag, dtype=A.dtype)
        return M_op


class JacobiPreconditionerGPU(PreconditionerABC):
    """
    Jacobi preconditioner: P = diag(A) on the GPU
    """
    def build(self, A: cps.spmatrix) -> cps.linalg.LinearOperator:
        """
        Builds the Jacobi preconditioner for sparse matrix A

        Args:
            A (cps.scipy.sparse.spmatrix): The system sparse matrix to precondition.

        Returns:
            LinearOperator representing the preconditioner (approximation of A^{-1})
        """
        diag = A.diagonal()
        M_op = cps.linalg.LinearOperator(shape=A.shape, matvec=lambda v: v / diag, dtype=A.dtype)
        return M_op


class SSORPreconditioner(PreconditionerABC):
    """
    Symmetric Successive Over-Relaxation (SSOR) preconditioner.
    """
    def __init__(self, omega: float = 1.0):
        """
        Initializes the SSOR preconditioner with relaxation factor omega.

        Args:
            omega (float): Relaxation factor (0 < omega < 2).
        """
        if not (0.0 < omega < 2.0):
            raise ValueError("SSOR requires 0 < omega < 2")
        self.omega = omega

    def build(self, A: sp.spmatrix) -> LinearOperator:
        """
        Builds the SSOR preconditioner for sparse matrix A

        Args:
            A (sp.spmatrix): The system sparse matrix to precondition.

        Returns:
            LinearOperator representing the preconditioner (approximation of A^{-1})
        """
        D = A.diagonal()

        L = tril(A, k=-1)
        U = triu(A, k=1)

        DL = L.copy()
        DL.setdiag(D / self.omega)

        DU = U.copy()
        DU.setdiag(D / self.omega)

        M_op = LinearOperator(
            shape=A.shape,
            matvec=lambda v: self.__matvec_ssor(v, DL, D, DU),
            dtype=A.dtype
            )
        return M_op

    @staticmethod
    def __matvec_ssor(v: npt.NDArray[np.float64], DL: sp.spmatrix, D: npt.NDArray[np.float64], DU: sp.spmatrix) -> npt.NDArray[np.float64]:
        """
        Perform the SSOR preconditioning operation M.

        Args:
            v (npt.NDArray[np.float64]): The input vector to precondition.
            DL (sp.spmatrix): The lower triangular part of the preconditioner.
            D (npt.NDArray[np.float64]): The diagonal of the preconditioner.
            DU (sp.spmatrix): The upper triangular part of the preconditioner.

        Returns:
            npt.NDArray[np.float64]: The preconditioned vector: M.
        """
        y = spsolve_triangular(DL, v, lower=True)
        z = y / D
        x = spsolve_triangular(DU, z, lower=False)
        return x


class ILUPreconditioner(PreconditionerABC):
    """
    Incomplete LU (ILU) preconditioner.
    """
    def __init__(self, drop_tol: float = 1e-4, fill_factor: float = 10):
        """
        Initializes the ILU preconditioner with drop tolerance and fill factor.

        Args:
            drop_tol (float): Drop tolerance for ILU factorization.
            fill_factor (float): Fill factor for ILU factorization.
        """
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor

    def build(self, A: sp.spmatrix) -> LinearOperator:
        """
        Builds the ILU preconditioner for sparse matrix A

        Args:
            A (sp.spmatrix): The system sparse matrix to precondition.

        Returns:
            LinearOperator representing the preconditioner (approximation of A^{-1})
        """

        ilu = spilu(A.tocsc(), drop_tol=self.drop_tol, fill_factor=self.fill_factor)

        M_op = LinearOperator(
            shape=A.shape,
            matvec = lambda v: ilu.solve(v),
            dtype=A.dtype
        )
        return M_op
