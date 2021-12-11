import numpy as np
from numpy.testing import assert_equal
from ufunclab import trapezoid_pulse


def test_basic():
    x = np.array([-4, -3, -2, -1, 0, 1, 1.5, 2, 3])
    y = trapezoid_pulse(x, -3, -1, 1, 2, 4)
    assert_equal(y, [0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 2.0, 0.0, 0.0])
