from enum import Enum
import numpy as np
from scipy.sparse import isspmatrix
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


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

def eigen_decomposition(M, K):
    """
    Perform eigen decomposition of a matrix.

    :param M: The mass matrix.
    :param K: The stiffness matrix.
    :return: A tuple containing the eigenvalues and eigenvectors.
    """

    f_max = 50
    nb_modes = int(M.shape[0] / 10)
    omega_max = 2 * np.pi * f_max
    sigma = (1.1 * omega_max) ** 2

    i = 0
    if isspmatrix(M):

        while True:
            # eigvals, eigvecs = eigsh(A=K, M=M, k=M.shape[0]-1, which='SM')
            eigvals, eigvecs = eigsh(A=K, M=M, k=nb_modes, sigma=0, which='LM', mode='normal')

            omega_values = np.sqrt(np.maximum(eigvals, 0.0))
            frequencies = omega_values / (2.0 * np.pi)
            idx_freq = frequencies <= f_max

            if frequencies[-1] < f_max:
                print("Warning: Not enough modes computed to cover the desired frequency range.")
                nb_modes = min(int(nb_modes * 2), M.shape[0] - 1)
                i += 1
            else:
                print("Done: Enough modes computed to cover the desired frequency range. took", i, "iterations.")
                break

        frequencies = frequencies[idx_freq]
        eigen_vectors = np.asarray(eigvecs[:, idx_freq])
        eigen_vals = eigvals[idx_freq]



        # X = np.random.rand(n, estimated_modes)
        # diag_K = K.diagonal()
        # diag_K_inv = np.where(np.abs(diag_K) > 1e-12, 1.0 / diag_K, 1.0)
        # M_pre = sparse.diags([diag_K_inv], [0], shape=(n, n))
        # eigvals, eigvecs = lobpcg(
        #         A=K,
        #         X=X,
        #         B=M,
        #         M=M_pre,
        #         largest=False,
        #         tol=1e-5,
        #         maxiter=1000
        #     )


    else:
        eigen_vals, eigen_vectors = eigh(A=K, M=M)
    return eigen_vals, eigen_vectors