import numpy as np
from numpy.testing import assert_allclose
from ufunclab import semivar


def test_semivar_exponential():
    y = semivar.exponential(1, 1, 3, 3)
    assert_allclose(y, 1 - 2*np.expm1(-1), rtol=1e-14)


def test_semivar_linear():
    y1 = semivar.linear(1, 2, 5, 3)
    assert_allclose(y1, 3.0, rtol=1e-14)
    y4 = semivar.linear(4, 2, 5, 3)
    assert y4 == 5


def test_semivar_spherical():
    y1 = semivar.spherical(1, 2, 5, 2)
    assert_allclose(y1, 65/16, rtol=1e-14)
    y4 = semivar.spherical(4, 2, 5, 2)
    assert y4 == 5
