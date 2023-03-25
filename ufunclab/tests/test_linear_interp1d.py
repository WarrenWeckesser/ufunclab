
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import linear_interp1d


def test_with_numpy_interp():
    xp = np.array([1, 2, 4, 8])
    fp = np.array([10, 14, 12, 2])
    x = np.array([1, 1.5, 2, 4.5, 8])
    yul = linear_interp1d(x, xp, fp)
    ynp = np.interp(x, xp, fp)
    assert_array_equal(yul, ynp)
