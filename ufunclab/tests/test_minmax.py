import pytest
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal
from ufunclab import minmax, min_argmin, max_argmax


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
