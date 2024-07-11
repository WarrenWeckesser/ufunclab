import pytest
import numpy as np
from numpy.testing import assert_equal
from ufunclab import nextn_less, nextn_greater


@pytest.mark.parametrize('func, to', [(nextn_less, -np.inf),
                                      (nextn_greater, np.inf)])
@pytest.mark.parametrize('dt', [np.dtype('float32'),
                                np.dtype('float64'),
                                np.dtype('longdouble')])
def test_nextn_less(func, to, dt):
    to = dt.type(to)
    x = dt.type(2.5)
    n = 5
    out = np.zeros(n, dtype=dt)
    xn = func(x, out=out)
    for k in range(n):
        x = np.nextafter(x, to)
        assert_equal(xn[k], x)
