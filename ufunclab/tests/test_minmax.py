import pytest
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal
from ufunclab import argmin, argmax, minmax, min_argmin, max_argmax


_numpy_types = [np.int8, np.uint8, np.int16, np.uint16,
                np.int32, np.uint32, np.int64, np.uint64,
                np.float32, np.float64, np.longdouble, object]


@pytest.mark.parametrize('dtype', _numpy_types)
@pytest.mark.parametrize('axis', [0, 1])
def test_argmin_basic(dtype, axis):
    x = np.array([[11, 10, 10, 23, 31],
                  [19, 20, 21, 22, 23]], dtype=dtype)
    i = argmin(x, axis=axis)
    assert_equal(i, np.argmin(x, axis=axis))


@pytest.mark.parametrize('dtype', _numpy_types)
@pytest.mark.parametrize('axis', [0, 1])
def test_argmax_basic(dtype, axis):
    x = np.array([[11, 10, 10, 23, 31],
                  [19, 20, 21, 22, 23]], dtype=dtype)
    i = argmax(x, axis=axis)
    assert_equal(i, np.argmax(x, axis=axis))


@pytest.mark.parametrize('x, expected', [
    (np.array([[1, 5, 2, 3], [3, 3, 3, 3]]),
     [[1, 5], [3, 3]]),
    (np.array([Fraction(2, 3), Fraction(8, 9), Fraction(-1, 3)]),
     [Fraction(-1, 3), Fraction(8, 9)]),
])
def test_minmax_basic(x, expected):
    mm = minmax(x)
    assert_equal(mm, expected)


@pytest.mark.parametrize('x, expected', [
    (np.array([[2, 5, 1, 4], [3, 3, 3, 3]]),
     ([1, 3], [2, 0])),
    (np.array([Fraction(2, 3), Fraction(8, 9), Fraction(-1, 3)]),
     (Fraction(-1, 3), 2)),
])
def test_min_argmin_basic(x, expected):
    m, argm = min_argmin(x)
    assert_equal(m, expected[0])
    assert_equal(argm, expected[1])


@pytest.mark.parametrize('x, expected', [
    (np.array([[2, 5, 1, 4], [3, 3, 3, 3]]),
     ([5, 3], [1, 0])),
    (np.array([Fraction(2, 3), Fraction(8, 9), Fraction(-1, 3)]),
     (Fraction(8, 9), 1)),
])
def test_max_argmax_basic(x, expected):
    m, argm = max_argmax(x)
    assert_equal(m, expected[0])
    assert_equal(argm, expected[1])


def test_minmax_length_zero_1d():
    x = np.array([])
    with pytest.raises(ValueError, match="n must be at least 1"):
        minmax(x)


def test_minmax_length_zero_2d():
    x = np.array([[], []])
    with pytest.raises(ValueError, match="n must be at least 1"):
        minmax(x, axes=[(1,), (0,)])
