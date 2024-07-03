import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ufunclab import backlash, backlash_sum


def test_all_zeros():
    x = np.zeros(10)
    y = backlash(x, 1, 0)
    assert_array_equal(y, x)


def test_within_deadband():
    x = np.array([0, 0, 0, 0.5, 0.5, 0.5, -0.75, -0.8])
    y = backlash(x, 2, 0)
    assert_array_equal(y, np.zeros_like(x))


def test_basic():
    x = np.array([0, 0, 1.5, 1.5, 1, 1, 1, -3, -3.25, -3])
    y = backlash(x, 2, 0)
    assert_array_equal(y, [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, -2, -2.25, -2.25])


def test_initial_a():
    x = np.array([1.25, 2, 3, 2.9, -1])
    y = backlash(x, 2, 0.5)
    assert_array_equal(y, [0.5, 1, 2, 2, 0])


def test_initial_b():
    x = np.array([1.75, 2, 3, 2.9, -1])
    y = backlash(x, 2, 0.5)
    assert_array_equal(y, [0.75, 1, 2, 2, 0])


def test_nonstandard_strides():
    t = np.linspace(0, 5, 11).reshape(-1, 1)
    y = (10*np.sin(t + [0, 1, np.pi/2])).T[::2]
    # y has strides (16, 24)
    # y.copy() has strides (88, 8)
    y1 = backlash(y, 1, 1, axis=1)
    y2 = backlash(y.copy(order='C'), 1, 1)
    assert_array_equal(y1, y2)


def test_out_nonstandard_strides():
    x = np.array([1, 2, 0, 3, 4, 5])
    base = np.zeros(2*len(x))
    out = base[::2]
    y1 = backlash(x, 1, 1)
    y2 = backlash(x, 1, 1, out=out)
    assert_array_equal(y1, y2)


def test_backlash_sum():
    x = np.array([0.25, 0.5, 10.0, 0.125, 0.2, 0.2, 0.2, 0.2])
    w = np.array([0.5, 3.0, 2.0])
    deadband = np.array([0.5, 1.0, 0.75])
    initial = np.array([1.5, 5.0, -2.0])
    yb = []
    for k in range(3):
        y = backlash(x, deadband[k], initial[k])
        yb.append(y)
    yb = np.array(yb)
    yy, final = backlash_sum(x, w, deadband, initial)
    assert_allclose(yy, w@yb, rtol=1e-14)
    assert_allclose(final, yb[:, -1], rtol=1e-14)
