
import pytest
import numpy as np
from numpy.testing import assert_equal
from fractions import Fraction
from ufunclab import searchsortedl, searchsortedr


@pytest.mark.parametrize('searchsorted', [searchsortedl, searchsortedr])
@pytest.mark.parametrize('v, expected', [(6, 5), (0, 0), (99, 8)])
@pytest.mark.parametrize('dtype', [np.int8, np.uint16,
                                   np.float16, np.float32, np.float64,
                                   object])
def test_basic(searchsorted, v, expected, dtype):
    sortedarr = np.array([1, 1, 2, 3, 5, 8, 13, 21], dtype=dtype)
    y = searchsorted(sortedarr, v)
    assert_equal(y, expected)


def test_object():
    sortedarr = np.array([Fraction(1, 9), Fraction(2, 9), Fraction(1, 2)])
    elements = np.array([Fraction(1, 6), Fraction(1, 2)])
    kl = searchsortedl(sortedarr, elements)
    assert_equal(kl, [1, 2])
    kr = searchsortedr(sortedarr, elements)
    assert_equal(kr, [1, 3])


def test_datetime64():
    sortedarr = np.array([np.datetime64('2001-01-02T19:18:17', 's'),
                          np.datetime64('2005-05-04T09:30:00', 's'),
                          np.datetime64('2005-12-01T15:45:00', 's')])
    elements = np.array([np.datetime64('2005-05-04T09:30', 'm'),
                         np.datetime64('2007-11-01T05:05', 'm')])
    kl = searchsortedl(sortedarr, elements)
    assert_equal(kl, [1, 3])
    kr = searchsortedr(sortedarr, elements)
    assert_equal(kr, [2, 3])


def test_timedelta64():
    sortedarr = np.array([[np.timedelta64(10, 'm'), np.timedelta64(100, 'm')],
                          [np.timedelta64(20, 'm'), np.timedelta64(100, 'm')],
                          [np.timedelta64(30, 'm'), np.timedelta64(200, 'm')]])
    elements = np.array([np.timedelta64(600, 's'), np.timedelta64(6000, 's')])
    kl = searchsortedl(sortedarr, elements[:, None])
    assert_equal(kl, [[0, 0, 0],
                      [1, 1, 1]])
    kr = searchsortedr(sortedarr, elements[:, None])
    assert_equal(kr, [[1, 0, 0],
                      [2, 2, 1]])
    kl0 = searchsortedl(sortedarr, elements, axis=0)
    assert_equal(kl0, [0, 0])
    kr0 = searchsortedr(sortedarr, elements, axis=0)
    assert_equal(kr0, [1, 2])
