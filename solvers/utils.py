from enum import Enum
import numpy as np
from scipy.sparse import isspmatrix


class LumpingMethod(Enum):
    """
    Enum class for the lumping methods.

    Based on :cite:p: `Zienkiewicz_2013`
    """
    RowSum = "RowSum"
    DiagonalScaling = "DiagonalScaling"
    MassProportion = "MassProportion"

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
        if self == LumpingMethod.MassProportion:
            return self.hrz(M_consistent)

    @staticmethod
    def row_sum(M_consistent):
        """
        Row-sum lumping method: Each diagonal entry is the sum of the corresponding row.

        :param M_consistent: The consistent mass matrix.
        :return: The lumped matrix.
        """
        if isspmatrix(M_consistent):
            M_lumped = np.sum(M_consistent, axis=1).A.ravel()
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

    @staticmethod
    def hrz(M_consistent):
        """
        Hinton-Rock-Zienkiewicz (HRZ) lumping: Row-sum with mass conservation scaling.

        :param M_consistent: The consistent matrix.
        :return: The lumped matrix.
        """
        M_total = M_consistent.sum()
        M_lumped = LumpingMethod.row_sum(M_consistent)
        lumped_mass_total = M_lumped.sum()
        scaling_factor = M_total / lumped_mass_total
        M_lumped *= scaling_factor
        return M_lumped
