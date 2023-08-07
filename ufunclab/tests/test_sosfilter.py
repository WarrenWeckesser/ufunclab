
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from ufunclab import sosfilter


exact_test_cases = [
    # sos, x, expected_y
    ([[1, 0.5, -0.75, 1, 0, 0]],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 0.5, -0.75, 0, 0, 0]),
    ([[1, 0.5, -0.75, 1, 0, 0],
      [1, 0.5, 0.25, 1, 0, 0]],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, -0.25, -0.25, -0.1875, 0]),
    ([[1, 0, 0, 1, -0.5, 0]],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 1, 0.5, 0.25, 0.125]),
    ([[1, 0, 2, 1, -1/2, 0]],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 1/2, 9/4, 9/8, 9/16]),
]


@pytest.mark.parametrize('dt', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('sos, x, expected_y', exact_test_cases)
def test_simple_float(dt, sos, x, expected_y):
    if dt == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    sos = np.array(sos, dtype=dt)
    x = np.array(x, dtype=dt)
    y = sosfilter(sos, x)
    assert_equal(y.dtype, dt)
    assert_array_equal(y, expected_y)
