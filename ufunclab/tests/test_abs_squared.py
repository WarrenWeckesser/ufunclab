import pytest
import numpy as np
from numpy.testing import assert_equal
from ufunclab import abs_squared, abssq


@pytest.mark.parametrize('dtype',
                         [np.float32, np.float64, np.longdouble])
def test_basic_real(dtype):
    if dtype == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    x = np.array([-1.0, 2.5, 9.0, 125.0], dtype=dtype)
    y = abs_squared(x)
    expected = x**2
    assert_equal(y.dtype, expected.dtype)
    assert_equal(y, expected)


@pytest.mark.parametrize('func', [abs_squared, abssq])
@pytest.mark.parametrize('dtype',
                         [np.complex64, np.complex128, np.clongdouble])
def test_basic_complex(func, dtype):
    if dtype == np.clongdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    z = np.array([-1-1j, 2.5j, 9.0-3j, 125.0, -8.0], dtype=dtype)
    y = func(z)
    expected = z.real**2 + z.imag**2
    assert_equal(y.dtype, expected.dtype)
    assert_equal(y, expected)
