import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import deadzone


def test_all_zeros():
    x = np.zeros(10)
    y = deadzone(x, -1, 1)
    assert_array_equal(y, x)


def test_basic():
    x = np.array([-4, -3, -2, -1, 0, 1, 2])
    y = deadzone(x, -2.5, 1.0)
    assert_array_equal(y, [-1.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
