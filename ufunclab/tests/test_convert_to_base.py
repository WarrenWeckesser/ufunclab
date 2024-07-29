
import numpy as np
from ufunclab import convert_to_base
from numpy.testing import assert_equal


def test_basic():
    x = 19
    base = 16
    digits = convert_to_base(x, base, ndigits=3)
    assert_equal(digits, [3, 1, 0])


def test_broadcasting():
    x = np.array([[2], [19], [255], [1000]])
    base = np.array([8, 16])
    digits = convert_to_base(x, base, ndigits=5)
    expected = np.array([[[2,   0, 0, 0, 0],
                          [2,   0, 0, 0, 0]],
                         [[3,   2, 0, 0, 0],
                          [3,   1, 0, 0, 0]],
                         [[7,   7, 3, 0, 0],
                          [15, 15, 0, 0, 0]],
                         [[0,   5, 7, 1, 0],
                          [8,  14, 3, 0, 0]]])
    assert_equal(digits, expected)


def test_axis():
    x = np.array([10, 13, 85])
    base = np.array([[8], [16]])
    d1 = convert_to_base(x, base, ndigits=4)
    d0 = convert_to_base(x, base, ndigits=4, axis=0)
    assert_equal(np.moveaxis(d0, 0, -1), d1)


def test_broadcasting_and_axis():
    x = np.array([[2], [19], [255], [1000]])
    base = np.array([8, 16])
    digits = convert_to_base(x, base, ndigits=5, axis=1)
    expected_default_axis = np.array([[[2,   0, 0, 0, 0],
                                       [2,   0, 0, 0, 0]],
                                      [[3,   2, 0, 0, 0],
                                       [3,   1, 0, 0, 0]],
                                      [[7,   7, 3, 0, 0],
                                       [15, 15, 0, 0, 0]],
                                      [[0,   5, 7, 1, 0],
                                       [8,  14, 3, 0, 0]]])
    assert_equal(digits, np.moveaxis(expected_default_axis, 1, -1))
