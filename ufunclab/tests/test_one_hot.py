import numpy as np
from numpy.testing import assert_equal
from ufunclab import one_hot


def test_basic_scalar():
    a = one_hot(3, 8)
    assert_equal(a, np.array([0, 0, 0, 1, 0, 0, 0, 0]))


def test_basic_1d_k():
    a = one_hot([3, 5, 6], 8)
    assert_equal(a, np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]]))
