
import pytest
from fractions import Fraction
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from ufunclab import peaktopeak


def test_simple_int8_1d():
    x = np.array([85, 125, 0, -75, -50], dtype=np.int8)
    p = peaktopeak(x)
    assert_equal(p, 200)
    assert_equal(p.dtype, np.uint8)


@pytest.mark.parametrize('dt', [np.float32, np.float64])
def test_simple_float(dt):
    z = np.array([1.5, 2.25, -3.5, 4.0, -1.0, -np.pi, 0.3],
                 dtype=dt)
    p = peaktopeak(z)
    assert_equal(p.dtype, dt)
    assert_equal(p, 7.5)


@pytest.mark.parametrize('dtype_in, dtype_out',
                         [(np.int8, np.uint8),
                          (np.uint8, np.uint8),
                          (np.int16, np.uint16),
                          (np.uint16, np.uint16),
                          (np.int32, np.uint32),
                          (np.uint32, np.uint32),
                          (np.int64, np.uint64),
                          (np.uint64, np.uint64)])
def test_simple_2d(dtype_in, dtype_out):
    x = np.array([[1, 3, 1, 10, 2],
                  [0, 9, 9, 10, 0]], dtype=dtype_in)
    p0 = peaktopeak(x, axis=0)
    assert_equal(p0.dtype, dtype_out)
    assert_array_equal(p0, np.array([1, 6, 8, 0, 2], dtype=dtype_out))
    p1 = peaktopeak(x, axis=1)
    assert_equal(p1.dtype, dtype_out)
    assert_array_equal(p1, np.array([9, 10], dtype=dtype_out))


def test_fractions_1d():
    x = np.array([Fraction(13, 6), Fraction(10, 9), Fraction(11, 3)],
                 dtype=object)
    p = peaktopeak(x)
    assert isinstance(p, Fraction)
    assert_equal(p, Fraction(23, 9))


def test_fractions_2d():
    x = np.array([[Fraction(13, 6), Fraction(10, 9), Fraction(11, 3)],
                  [Fraction(12, 7), Fraction(12, 7), Fraction(12, 7)]],
                 dtype=object)
    p0 = peaktopeak(x, axis=0)
    assert_equal(p0.dtype, np.dtype(object))
    assert_array_equal(p0, [Fraction(19, 42),
                            Fraction(38, 63),
                            Fraction(41, 21)])
    p1 = peaktopeak(x, axis=1)
    assert_equal(p0.dtype, np.dtype(object))
    assert_array_equal(p1, [Fraction(23, 9), Fraction(0)])


# Removed until datetime64 and timedelta64 handling is restored
# in ufunclab.peaktopeak.
#
# def test_dates():
#     dates = np.array([np.datetime64('2015-11-02T12:34:50'),
#                       np.datetime64('2015-11-02T10:00:00'),
#                       np.datetime64('2015-11-02T21:20:19'),
#                       np.datetime64('2015-11-02T19:25:00')])
#     timespan = peaktopeak(dates)
#     assert_equal(timespan.dtype, np.dtype('m8[s]'))
#     assert_equal(timespan, np.timedelta64(40819, 's'))
