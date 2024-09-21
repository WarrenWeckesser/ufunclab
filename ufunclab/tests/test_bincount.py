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
