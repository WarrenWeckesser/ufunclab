import pytest
import numpy as np
from numpy.testing import assert_equal
from ufunclab import bincount


def test_1d_default_m():
    x = np.array([0, 0, 3, 4, 3, 4, 0, 3, 0])
    y = bincount(x)
    assert_equal(y, [4, 0, 0, 3, 2])


def test_1d_given_m():
    x = np.array([0, 0, 3, 4, 3, 4, 0, 3, 0])
    y = bincount(x, 8)
    assert_equal(y, [4, 0, 0, 3, 2, 0, 0, 0])


@pytest.mark.parametrize('m', [None, 2, 5, 8])
def test_nd_default_m(m):
    x = np.array([[0, 4, 4, 3, 2, 1],
                  [1, 1, 2, 2, 3, 3],
                  [4, 4, 4, 4, 4, 2]])
    ref = np.array([[1, 1, 1, 1, 2],
                    [0, 2, 2, 2, 0],
                    [0, 0, 1, 0, 5]])
    refm = max(np.max(x) + 1, 0)
    y = bincount(x, m=m)
    if (m is None):
        expected = ref
    elif m <= refm:
        expected = ref[:, :m]
    else:
        expected = np.pad(ref, [(0, 0), (0, m - refm)])
    assert_equal(y, expected)


@pytest.mark.parametrize('m', [None, 2, 5, 8])
def test_axis(m):
    x = np.array([[0, 4, 4],
                  [1, 1, 2],
                  [4, 4, 4],
                  [0, 0, 2]])
    ref = np.array([[2, 1, 0],
                    [1, 1, 0],
                    [0, 0, 2],
                    [0, 0, 0],
                    [1, 2, 2]])
    refm = max(np.max(x) + 1, 0)
    y = bincount(x, m=m, axis=0)
    if (m is None):
        expected = ref
    elif m <= refm:
        expected = ref[:m, :]
    else:
        expected = np.pad(ref, [(0, m - refm), (0, 0)])
    assert_equal(y, expected)


def test_weights():
    x = np.array([3, 1, 1, 0, 3])
    w = np.array([4, 9, 1, 3, 5])
    b = bincount(x, weights=w)
    assert_equal(b, [3, 10, 0, 9])


def test_weights_axis():
    x = np.array([[3, 1, 1],
                  [2, 3, 3],
                  [1, 2, 2],
                  [2, 2, 3]])
    w = np.array([2.0, 3.0, 5.0, 4.0])
    expected = np.array([[0.0, 0.0, 0.0],
                         [5.0, 2.0, 2.0],
                         [7.0, 9.0, 5.0],
                         [2.0, 3.0, 7.0]])
    b = bincount(x, weights=w, axis=0)
    assert_equal(b, expected)


@pytest.mark.parametrize('dtype', [np.dtype('F'), np.dtype('D')])
def test_complex_weights(dtype):
    x = np.array([3, 1, 1, 0, 3])
    w = np.array([4+1j, 9-1j, 1+0.5j, 3+0.25j, 5.0], dtype=dtype)
    b = bincount(x, weights=w)
    assert b.dtype == w.dtype
    assert_equal(b, [3 + 0.25j, 10 - 0.5j, 0.0, 9.0 + 1.0j])


def test_complex_weights_axis():
    x = np.array([[3, 1, 1],
                  [2, 3, 3],
                  [1, 2, 2],
                  [2, 2, 3]])
    w = np.array([2.0 + 2j, 3.0 + 7j, 5.0, 4.0 - 1j])
    b = bincount(x, weights=w, axis=0)
    br = bincount(x, weights=w.real, axis=0)
    bi = bincount(x, weights=w.imag, axis=0)
    expected_br = np.array([[0.0, 0.0, 0.0],
                            [5.0, 2.0, 2.0],
                            [7.0, 9.0, 5.0],
                            [2.0, 3.0, 7.0]])
    expected_bi = np.array([[0.0,  0.0, 0.0],
                            [0.0,  2.0, 2.0],
                            [6.0, -1.0, 0.0],
                            [2.0,  7.0, 6.0]])
    assert_equal(br, expected_br)
    assert_equal(bi, expected_bi)
    assert_equal(b, br + 1j*bi)
