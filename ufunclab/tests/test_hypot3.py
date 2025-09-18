import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from ufunclab import hypot3


@pytest.mark.parametrize('x, y, z', [(1.0, 2.0, -3.0),
                                     (1.0, [2.0, 3.0], [4.0, 5.0])])
def test_hypot3_basic(x, y, z):
    h = hypot3(x, y, z)
    ref = np.hypot(x, np.hypot(y, z))
    assert_allclose(h, ref, rtol=5e-15)


@pytest.mark.parametrize('x, y, z', [(-2.5, np.inf, 99.0),
                                     (np.nan, 5.5, np.inf)])
def test_hypot3_inf(x, y, z):
    assert_equal(hypot3(x, y, z), np.inf)


def test_hypot3_nan_input():
    x = np.array([3.0, 0.125])
    y = np.array([5.0])
    z = np.array([0.0, np.nan])
    h = hypot3(x, y, z)
    ref = np.hypot(x, np.hypot(y, z))
    assert_allclose(h, ref, rtol=5e-15)
