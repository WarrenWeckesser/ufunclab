import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import backlash


def test_all_zeros():
    x = np.zeros(10)
    y = backlash(x, 1, 0)
    assert_array_equal(y, x)


def test_within_deadband():
    x = np.array([0, 0, 0, 0.5, 0.5, 0.5, -0.75, -0.8])
    y = backlash(x, 2, 0)
    assert_array_equal(y, np.zeros_like(x))


def test_basic():
    x = np.array([0, 0, 1.5, 1.5, 1, 1, 1, -3, -3.25, -3])
    y = backlash(x, 2, 0)
    assert_array_equal(y, [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, -2, -2.25, -2.25])


def test_initial_a():
    x = np.array([1.25, 2, 3, 2.9, -1])
    y = backlash(x, 2, 0.5)
    assert_array_equal(y, [0.5, 1, 2, 2, 0])


def test_initial_b():
    x = np.array([1.75, 2, 3, 2.9, -1])
    y = backlash(x, 2, 0.5)
    assert_array_equal(y, [0.75, 1, 2, 2, 0])
