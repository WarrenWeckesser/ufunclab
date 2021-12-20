import pytest
import numpy as np
from numpy.testing import assert_equal

# The 'wrappers' code is currently experimental, and not yet
# an importable package.  To run these tests with pytest,
# add the current directory to the Python search path.
from wrappers import all_same, deadzone, minmax, vnorm


@pytest.mark.parametrize('axis, expected', [
        (None, False),
        (0, [True, False, False, False]),
        (1, [False, False, True]),
        ((1, 0), False)
])
def test_all_same_axis(axis, expected):
    x = np.array([[1, 2, 3, 4],
                  [1, 7, 8, 9],
                  [1, 1, 1, 1]])
    y = all_same(x, axis=axis)
    assert_equal(y, expected)


def test_all_same_keepdims():
    x = np.array([[1, 2, 3, 4],
                  [1, 7, 8, 9],
                  [1, 1, 1, 1]])
    y = all_same(x, axis=1, keepdims=True)
    assert_equal(y, [[False], [False], [True]])


def test_deadzone_keywords():
    x = np.array([[-1, 0, 1, 2, 3, 4, 5],
                  [-3, 2, 4, 6, 8, 9, 10]])
    y = deadzone(x, high=2, low=0)
    assert_equal(y, [[-1, 0, 0, 0, 1, 2, 3],
                     [-3, 0, 2, 4, 6, 7, 8]])


def test_deadzone_where():
    x = np.array([[-1, 0, 1, 2, 3, 4, 5],
                  [-3, 2, 4, 6, 8, 5, 3]])
    out = np.full_like(x, fill_value=9, dtype=np.float64)
    y = deadzone(x, low=-0.5, high=2, where=x <= 1, out=out)
    assert y is out
    assert_equal(y, [[-0.5, 0, 0, 9, 9, 9, 9],
                     [-2.5, 9, 9, 9, 9, 9, 9]])


def test_minmax_basic():
    x = np.array([[2, 0, 4, 5, 3, 2, 1],
                  [1, 1, 1, 2, 2, 2, 2],
                  [0, 9, 0, 9, 0, 2, 4]])
    y = minmax(x)
    assert_equal(y, [[0, 5], [1, 2], [0, 9]])


def test_minmax_axes():
    x = np.array([[2, 0, 4, 5, 3, 2, 1],
                  [1, 1, 1, 2, 2, 2, 2],
                  [0, 9, 0, 9, 0, 2, 4]])
    y = minmax(x, axes=[(0,), (1,)])
    assert_equal(y, [[0, 2], [0, 9], [0, 4], [2, 9], [0, 3], [2, 2], [1, 4]])


@pytest.mark.parametrize('transpose', [False, True])
def test_vnorm_default_p(transpose):
    x = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [1, 2, 2],
                  [3, 4, 0]])
    if transpose:
        m = vnorm(x.T, axis=0)
    else:
        m = vnorm(x)
    # Using equal with floating point is risky...
    assert_equal(m, [0, np.sqrt(3), 3, 5])


def test_vnorm_p_keyword_and_broadcasting():
    x = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [1, 2, 2],
                  [3, 4, 0]])
    y = vnorm(np.expand_dims(x, 2), axis=0, p=[1, np.inf])
    assert_equal(y, [[5.0, 3.0], [7.0, 4.0], [3.0, 2.0]])
