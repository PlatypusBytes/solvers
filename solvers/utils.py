from enum import Enum
from typing import Optional, Union
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse import isspmatrix


class LumpingMethod(Enum):
    """
    Enum class for the lumping methods.

    Based on :cite:p: `Zienkiewicz_2013`
    """
    RowSum = "RowSum"
    DiagonalScaling = "DiagonalScaling"
    NONE = "None"

    def apply(self, M_consistent: Union[sp.spmatrix, npt.NDArray[np.float64]]) -> Optional[npt.NDArray[np.float64]]:
        """
        Apply the selected lumping method to the input consistent mass matrix.

        Args:
            M_consistent (Union[sp.spmatrix, npt.NDArray[np.float64]]): The consistent mass matrix to be lumped.

        Returns:
            Optional[npt.NDArray[np.float64]]: The lumped mass matrix as a 1D array (vector) of diagonal values,
            or None if no lumping is applied.
        """
        if self == LumpingMethod.RowSum:
            return self.row_sum(M_consistent)
        if self == LumpingMethod.DiagonalScaling:
            return self.diagonal_scaling(M_consistent)
        if self == LumpingMethod.NONE:
            return None

    @staticmethod
    def row_sum(M_consistent: Union[sp.spmatrix, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Row-sum lumping method: Each diagonal entry is the sum of the corresponding row.

        Args:
            M_consistent (Union[sp.spmatrix, npt.NDArray[np.float64]]): The consistent mass matrix to be lumped.

        Returns:
            npt.NDArray[np.float64]: The lumped mass matrix as a 1D array (vector) of diagonal values.
        """
        if isspmatrix(M_consistent):
            M_lumped = np.array(M_consistent.sum(axis=1)).ravel()
        else:
            M_lumped = np.sum(M_consistent, axis=1)
        return M_lumped

    @staticmethod
    def diagonal_scaling(M_consistent: Union[sp.spmatrix, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Diagonal scaling lumping: Distributes total mass proportionally to the diagonal entries.

        Args:
            M_consistent (Union[sp.spmatrix, npt.NDArray[np.float64]]): The consistent mass matrix to be lumped.

        Returns:
            npt.NDArray[np.float64]: The lumped mass matrix as a 1D array (vector) of diagonal values.
        """
        M_total = M_consistent.sum()
        diag_sum = M_consistent.diagonal().sum()
        scale_factor = M_total / diag_sum
        M_lumped = M_consistent.diagonal() * scale_factor
        return M_lumped
