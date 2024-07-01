
import pytest
import numpy as np
from numpy.testing import assert_equal
from fractions import Fraction
from ufunclab import cross3, cross2


# In numpy 2.0, the handling of length-2 vectors in np.cross
# is deprecated.  To avoid the deprecation warning, this function
# is used as a wrapper of np.cross.  This wrapper does not provide
# the parameter `axisc`.
def numpy_cross2(a, b, axisa=-1, axisb=-1):
    a = np.asarray(a)
    b = np.asarray(b)
    apad = [(0, 0)]*a.ndim
    apad[axisa] = (0, 1)
    bpad = [(0, 0)]*b.ndim
    bpad[axisb] = (0, 1)
    a3 = np.pad(a, apad)
    b3 = np.pad(b, bpad)
    c = np.cross(a3, b3, axisa=axisa, axisb=axisb)
    return c[..., -1]


@pytest.mark.parametrize('u, v', [([1, 2, 3], [5, 3, 1]),
                                  ([1.5, 0.5, -1.5], [2.0, 9.0, -3.0])])
def test_cross3_basic(u, v):
    w = cross3(u, v)
    assert_equal(w, np.cross(u, v))


def test_cross3_object():
    u = np.array([1, 2, Fraction(1, 2)], dtype=object)
    v = np.array([-2.5, 8, Fraction(3, 2)], dtype=object)
    w = cross3(u, v)
    assert w.dtype == np.dtype(object)
    assert_equal(w, np.array([-1, -2.75, 13.0], dtype=object))


def test_cross3_basic_broadcasting():
    u = np.arange(21).reshape(7, 1, 3)
    v = -0.5*np.arange(15).reshape(1, 5, 3)
    w = cross3(u, v)
    assert_equal(w, np.cross(u, v))


def test_cross3_nontrivial_axes():
    u = np.arange(12).reshape(4, 3, 1)
    v = np.arange(6).reshape(3, 1, 2)
    w = cross3(u, v, axes=[1, 0, 2])
    assert_equal(w, np.cross(u, v, axisa=1, axisb=0, axisc=2))


@pytest.mark.parametrize('u, v', [([1, 2], [5, 3]),
                                  ([1.5, 0.5], [2.0, 9.0])])
def test_cross2_basic(u, v):
    w = cross2(u, v)
    assert_equal(w, numpy_cross2(u, v))


def test_cross2_broadcasting():
    x = np.arange(70).reshape(7, 2, 5)
    y = np.arange(10).reshape(5, 2)
    z = cross2(x, y, axes=[(1,), (1,)])
    assert_equal(z, numpy_cross2(x, y, axisa=1, axisb=1))


def test_cross2_object():
    u = np.array([Fraction(2, 3), Fraction(1, 5)])
    v = np.array([[Fraction(3, 2), Fraction(3, 5)], [8, 9]])
    w = cross2(u, v)
    expected = np.array([Fraction(1, 10), Fraction(22, 5)])
    assert w.dtype == np.dtype(object)
    assert_equal(w, expected)
