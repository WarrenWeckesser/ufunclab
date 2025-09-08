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


@pytest.mark.parametrize('val', [np.nan, np.inf])
def test_nan_or_inf(val):
    x = np.array([0.75, -0.125, 0.0, val, 0.25, 0.55])
    y = unwrap(x, 1.0)
    assert_array_equal(y, np.array([0.75, 0.875, 1.0, np.nan, np.nan, np.nan]))
