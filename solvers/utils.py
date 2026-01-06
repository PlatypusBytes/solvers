from enum import Enum
import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, spsolve_triangular, spilu
from scipy.sparse import tril, triu


class PreConditioner(Enum):
    """
    Enum class for the preconditioner types.
    """
    NONE = "None"
    JACOBI = "Jacobi"
    SSOR = "SSOR"
    ILU = "ILU"


    def apply(self, A, **kwargs):
        """
        Returns a LinearOperator representing M^{-1}

        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            The system matrix to precondition.
        kwargs : preconditioner-specific parameters
        """

        if self == PreConditioner.NONE:
            return None

        if self == PreConditioner.JACOBI:
            return self.__jacobi(A)

        if self == PreConditioner.SSOR:
            return self.__ssor(A, **kwargs)

        if self == PreConditioner.ILU:
            return self.__ilu(A, **kwargs)

        raise ValueError(f"Unknown preconditioner {self}")

    def __jacobi(self, A):
        """
        Jacobi preconditioner: M = diag(A)

        Parameters
        ----------
        :param A: The system matrix.
        :return: LinearOperator representing M^{-1}
        """
        diag = A.diagonal()
        M_op = LinearOperator(shape=A.shape, matvec=lambda v: v / diag, dtype=A.dtype)
        return M_op

    def __ssor(self, A, omega=1.0):
        """
        Symmetric Successive Over-Relaxation (SSOR) preconditioner.

        Parameters
        ----------
        :param A: The system matrix.
        :param omega: Relaxation factor (0 < omega < 2).
        :return: LinearOperator representing M^{-1}
        """

        if not (0.0 < omega < 2.0):
            raise ValueError("SSOR requires 0 < omega < 2")

        D = A.diagonal()

        L = tril(A, k=-1)
        U = triu(A, k=1)

        DL = L.copy()
        DL.setdiag(D / omega)

        DU = U.copy()
        DU.setdiag(D / omega)

        M_op = LinearOperator(
            shape=A.shape,
            matvec=lambda v: self.__matvec_ssor(v, DL, D, DU),
            dtype=A.dtype
            )
        return M_op

    def __ilu(self, A, drop_tol=1e-4, fill_factor=10):
        """
        Incomplete LU (ILU) preconditioner.

        Parameters
        ----------
        :param A: The system matrix.
        :param drop_tol: Drop tolerance for ILU.
        :param fill_factor: Fill factor for ILU.
        :return: LinearOperator representing M^{-1}
        """

        ilu = spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)

        M_op = LinearOperator(
            shape=A.shape,
            matvec = lambda v: ilu.solve(v),
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

class LumpingMethod(Enum):
    """
    Enum class for the lumping methods.

    Based on :cite:p: `Zienkiewicz_2013`
    """
    RowSum = "RowSum"
    DiagonalScaling = "DiagonalScaling"
    NONE = "None"

    def apply(self, M_consistent):
        """
        Apply the selected lumping method to the input consistent mass matrix.

        :param M_consistent: The consistent mass matrix.
        :return: The lumped matrix.
        """
        if self == LumpingMethod.RowSum:
            return self.row_sum(M_consistent)
        if self == LumpingMethod.DiagonalScaling:
            return self.diagonal_scaling(M_consistent)
        if self == LumpingMethod.NONE:
            return None

    @staticmethod
    def row_sum(M_consistent):
        """
        Row-sum lumping method: Each diagonal entry is the sum of the corresponding row.

        :param M_consistent: The consistent mass matrix.
        :return: The lumped matrix as a 1D array (vector) of diagonal values.
        """
        if isspmatrix(M_consistent):
            M_lumped = np.array(M_consistent.sum(axis=1)).ravel()
        else:
            M_lumped = np.sum(M_consistent, axis=1)
        return M_lumped

    @staticmethod
    def diagonal_scaling(M_consistent):
        """
        Diagonal scaling lumping: Distributes total mass proportionally to the diagonal entries.

        :param M_consistent: The consistent mass matrix.
        :return: The lumped matrix.
        """
        M_total = M_consistent.sum()
        diag_sum = M_consistent.diagonal().sum()
        scale_factor = M_total / diag_sum
        M_lumped = M_consistent.diagonal() * scale_factor
        return M_lumped
