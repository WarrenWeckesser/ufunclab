
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import next_greater, next_less


@pytest.mark.parametrize('func, to', [(next_greater, np.inf),
                                      (next_less, -np.inf)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_next_greater(func, to, dtype):
    if dtype == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    x = np.array([-np.inf, -1234.5, -1.0, 0, 1.0, 1000, np.inf, np.nan],
                 dtype=dtype)
    y = func(x)
    na = np.nextafter(x, dtype(to))
    assert_array_equal(y, na, strict=True)
