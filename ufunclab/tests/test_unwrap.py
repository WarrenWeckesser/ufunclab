import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import unwrap



def test_basic():
    x = [7.0, 0.25, 13.5]
    period = 4.0
    y = unwrap(x, period)
    assert_array_equal(y, np.array([7.0, 8.25, 9.5]))


@pytest.mark.parametrize(
    'x, period',
    [([0.25, 3, 4, 9, 10], 4.25),
     ([-10, 25, 0.375, 0.125], 1.5)]
)
def test_against_numpy_unwrap(x, period):
    y = unwrap(x, period)
    assert_array_equal(y, np.unwrap(x, period=period))
    xr = x[::-1]
    yr = unwrap(xr, period)
    assert_array_equal(yr, np.unwrap(xr, period=period))
