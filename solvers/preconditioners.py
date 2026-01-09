from abc import ABC, abstractmethod

import scipy.sparse as sp
from scipy.sparse import tril, triu
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve_triangular, spilu


class PreconditionerABC(ABC):
    """
    Abstract base class for preconditioners
    """
    @abstractmethod
    def build(self, A: sp.spmatrix):
        """
        Build the preconditioner for sparse matrix A

        Parameters
        ----------
        A : sp.spmatrix
            The system matrix to precondition.
        """
        raise NotImplementedError("Subclasses should implement this!")

class JacobiPreconditioner(PreconditionerABC):
    """
    Jacobi preconditioner: M = diag(A)
    """
    def build(self, A: sp.spmatrix) -> LinearOperator:
        """
        Builds the Jacobi preconditioner for sparse matrix A

        Parameters
        ----------
        A : sp.spmatrix
            The system sparse matrix to precondition.
        Returns
        -------
        LinearOperator representing M^{-1}
        """
        diag = A.diagonal()
        M_op = LinearOperator(shape=A.shape, matvec=lambda v: v / diag, dtype=A.dtype)
        return M_op


class SSORPreconditioner(PreconditionerABC):
    """
    Symmetric Successive Over-Relaxation (SSOR) preconditioner.
    """
    def __init__(self, omega: float = 1.0):
        """
        Initializes the SSOR preconditioner with relaxation factor omega.

        Parameters
        ----------
        omega : Relaxation factor (0 < omega < 2).
        """
        if not (0.0 < omega < 2.0):
            raise ValueError("SSOR requires 0 < omega < 2")
        self.omega = omega

    def build(self, A: sp.spmatrix) -> LinearOperator:
        """
        Builds the SSOR preconditioner for sparse matrix A

        Parameters
        ----------
        A : sp.spmatrix
            The system sparse matrix to precondition.
        Returns
        -------
        LinearOperator representing M^{-1}
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
    def __matvec_ssor(v, DL, D, DU):
        """
        Perform the SSOR preconditioning operation M^{-1} * v.

        Parameters
        ----------
        :param v: The input vector.
        :param DL: Lower triangular matrix with modified diagonal.
        :param D: Diagonal entries.
        :param DU: Upper triangular matrix with modified diagonal.
        :return: The result of M^{-1} * v.
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

        Parameters
        ----------
        drop_tol : Drop tolerance for ILU.
        fill_factor : Fill factor for ILU.
        """
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor

    def build(self, A: sp.spmatrix) -> LinearOperator:
        """
        Builds the ILU preconditioner for sparse matrix A

        Parameters
        ----------
        A : sp.spmatrix
            The system sparse matrix to precondition.
        Returns
        -------
        LinearOperator representing M^{-1}
        """

        ilu = spilu(A.tocsc(), drop_tol=self.drop_tol, fill_factor=self.fill_factor)

        M_op = LinearOperator(
            shape=A.shape,
            matvec = lambda v: ilu.solve(v),
            dtype=A.dtype
        )
        return M_op
