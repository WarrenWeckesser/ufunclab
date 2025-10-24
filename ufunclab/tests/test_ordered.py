import pytest
from fractions import Fraction
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import ordered, op


@pytest.mark.parametrize('typ', [np.int8, np.uint8,
                                 np.int16, np.uint16,
                                 np.int32, np.uint32,
                                 np.int64, np.uint64,
                                 np.float32, np.float64])
def test_constant_input(typ):
    x = np.ones(8, dtype=typ)
    assert_array_equal(ordered(x, [op.LT, op.LE, op.EQ, op.GE, op.GT]),
                       [False, True, True, True, False])


def test_basic():
    x = np.array([20, 15, 10, 10, 9, 3])
    assert_array_equal(ordered(x, [op.LT, op.LE, op.EQ, op.GE, op.GT]),
                       [False, False, False, True, False])


def test_with_axis():
    x = np.array([[1, 2, 4],
                  [1, 1, 3],
                  [1, 1, 2],
                  [1, 1, 1]])

    assert_array_equal(ordered(x, op.LT), [True, False, False, False])
    assert_array_equal(ordered(x, op.LE), [True, True, True, True])
    assert_array_equal(ordered(x, op.GE), [False, False, False, True])

    assert_array_equal(ordered(x, op.EQ, axis=0), [True, False, False])
    assert_array_equal(ordered(x, op.GT, axis=0), [False, False, True])


def test_object_array():
    x = np.array([10, Fraction(355, 113), Fraction(5, 2), Fraction(1, 3)])
    q = ordered(x, [op.GT, op.GE, op.EQ])
    assert_array_equal(q, [True, True, False])
