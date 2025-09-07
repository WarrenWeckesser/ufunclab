import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import vdot


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_single(dtype):
    if dtype == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    a = np.array([1, 2, 3, 0, 4, 4, 4], dtype=dtype)
    b = np.array([2, 1, 4, 3, 1, 1, 2], dtype=dtype)
    d = vdot(a, b)
    assert isinstance(d, dtype)
    assert d == 32.0


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_basic_arrays(dtype):
    if dtype == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    a = np.array([[1, 2, 3, 0, 4, 4, 4],
                  [1, 1, 1, 1, 2, 2, 2]], dtype=dtype)
    b = np.array([[2, 1, 4, 3, 1, 1, 2],
                  [0, 0, 0, 1, 1, 1, 2]], dtype=dtype)
    d = vdot(a, b)
    assert_array_equal(d, np.array([32.0, 9.0], dtype=dtype), strict=True)

    d0 = vdot(a, b, axis=0)
    assert_array_equal(d0, np.array([2.0, 2.0, 12.0, 1.0, 6.0, 6.0, 12.0],
                                    dtype=dtype), strict=True)
