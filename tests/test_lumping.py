
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from solvers.utils import LumpingMethod

@pytest.fixture
def M_consistent():
    """
    Fixture to provide a consistent mass matrix for testing.
    """
    return np.array([
        [2.0, 0, 1.0, 0],
        [0, 2.0, 1.0, 1.0],
        [1.0, 0, 2.0, 0],
        [1.0, 1.0, 0, 2.0]
    ])

class TestLumpedMatrix:

    def test_enums(self):
        """
        Test the LumpingMethod enum values.
        """
        assert LumpingMethod.RowSum.value == "RowSum"
        assert LumpingMethod.DiagonalScaling.value == "DiagonalScaling"

    def test_row_sum_lumping(self, M_consistent):
        """
        Test the row-sum lumping method.
        """
        M_lumped = LumpingMethod.RowSum.apply(M_consistent)
        M_lumped_expected = np.array([3, 4., 3., 4.])
        np.testing.assert_array_almost_equal(M_lumped, M_lumped_expected)
        # test that the mass is constant after lumping
        assert np.sum(M_lumped) == np.sum(M_consistent)

    def test_row_sum_lumping_sparse(self, M_consistent):
        """
        Test the row-sum lumping method with sparse matrix.
        """
        M_consistent_sparse = csr_matrix(M_consistent)

        M_lumped = LumpingMethod.RowSum.apply(M_consistent_sparse)
        M_lumped_expected = np.array([3, 4., 3., 4.])
        np.testing.assert_array_almost_equal(M_lumped, M_lumped_expected)
        # test that the mass is constant after lumping
        assert np.sum(M_lumped) == np.sum(M_consistent)


    def test_diagonal_scaling(self, M_consistent):
        """
        Test the diagonal scaling lumping method.
        """
        M_lumped = LumpingMethod.DiagonalScaling.apply(M_consistent)
        M_lumped_expected = np.array([3.5, 3.5, 3.5, 3.5])
        np.testing.assert_array_almost_equal(M_lumped, M_lumped_expected)
        # test that the mass is constant after lumping
        assert np.sum(M_lumped) == np.sum(M_consistent)

    def test_diagonal_scaling_sparse(self, M_consistent):
        """
        Test the diagonal scaling lumping method.
        """
        M_consistent_sparse = csr_matrix(M_consistent)
        M_lumped = LumpingMethod.DiagonalScaling.apply(M_consistent_sparse)
        M_lumped_expected = np.array([3.5, 3.5, 3.5, 3.5])
        np.testing.assert_array_almost_equal(M_lumped, M_lumped_expected)
        # test that the mass is constant after lumping
        assert np.sum(M_lumped) == np.sum(M_consistent)
