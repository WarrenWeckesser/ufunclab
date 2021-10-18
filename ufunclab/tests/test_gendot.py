
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


@pytest.mark.xfail(reason="need to deal with type resolution in prodfunc",
                   raises=TypeError)
def test_datetime_timedelta_add_max():
    addmax = gendot(np.add, np.maximum)
    a = np.array([np.datetime64('2021-01-01T12:55:55'),
                  np.datetime64('2021-01-01T13:00:00')])
    b = np.array([np.timedelta64(60, 's'), np.timedelta64(-360, 's')])
    c = addmax(a, b)
    assert c.dtype == a.dtype
    assert_equal(c, np.maximum.reduce(np.add(a, b)))


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128,
                                   np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.int8, np.int16, np.int32, np.int64])
def test_identity(dtype):
    am = gendot(np.add, np.multiply)
    a = np.array([], dtype=dtype)
    b = np.array([], dtype=dtype)
    p = am(a, b)
    assert p.dtype == dtype
    assert p == np.multiply.identity


def test_no_identity():
    # The ufunc np.maximum does not have an identity element.
    minmaxdot = gendot(np.minimum, np.maximum)
    with pytest.raises(ValueError, match='with no identity'):
        minmaxdot([], [])
