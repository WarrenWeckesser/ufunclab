
import pytest
import numpy as np
from numpy.testing import assert_equal
from ufunclab import gendot


def test_minmaxdot_1d():
    minmaxdot = gendot(np.minimum, np.maximum)
    a = np.array([1, 3, 1, 9, 1, 2])
    b = np.array([2, 0, 5, 1, 3, 2])
    c = minmaxdot(a, b)
    assert c == np.maximum.reduce(np.minimum(a, b))


def test_minmaxdot_broadcasting():
    minmaxdot = gendot(np.minimum, np.maximum)
    rng = np.random.default_rng(39923480898981)
    x = rng.exponential(3, size=(10, 1000))
    y = rng.exponential(3, size=(5, 1, 1000))
    z = minmaxdot(x, y)
    assert_equal(z, np.maximum.reduce(np.minimum(x, y), axis=-1))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_and_or(dtype):
    bitwise_and_or = gendot(np.bitwise_and, np.bitwise_or)
    a = np.array([11, 41, 15, 11, 20, 14, 21], dtype=dtype)
    b = np.array([[51, 13, 18, 43, 12, 71, 47],
                  [14, 13, 28, 33, 87, 31, 79]], dtype=dtype)
    c = bitwise_and_or(a, b)
    expected = np.bitwise_or.reduce(np.bitwise_and(a, b), axis=-1)
    assert c.dtype == expected.dtype
    assert_equal(c, expected)
