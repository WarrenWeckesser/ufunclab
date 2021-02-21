import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import hysteresis_relay




def test_basic():
    x = np.array([-4, -3, -2, -1, 0, 1, 2, 1, 0, -2])
    y = hysteresis_relay(x, -0.5, 0.5, -1, 1, -1)
    assert_array_equal(y, [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1])


def test_init():
    x = np.array([0, 0, 9, 0, -9, 9])
    y = hysteresis_relay(x, -1, 1, -2, 2, -3)
    assert_array_equal(y, [-3, -3, 2, 2, -2, 2])
