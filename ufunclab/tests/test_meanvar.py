
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from ufunclab import meanvar


@pytest.mark.parametrize('ddof', [0, 1])
@pytest.mark.parametrize('transpose', [False, True])
def test_basic(ddof, transpose):
    x = np.array([[1, 4, 4, 2, 1, 1, 2, 7],
                  [0, 0, 9, 4, 1, 0, 0, 1],
                  [8, 3, 3, 3, 3, 3, 3, 3],
                  [5, 5, 5, 5, 5, 5, 5, 5]])
    if transpose:
        x = x.T
    mv = meanvar(x, ddof)
    assert_equal(mv[:, 0], np.mean(x, axis=-1))
    assert_allclose(mv[:, 1], np.var(x, ddof=ddof, axis=-1),
                    rtol=5e-16)


def test_out_with_strides():
    x = np.array([1, 1, 3, 1])
    ddof = [0, 1, 2]
    a = np.zeros((5, 3), dtype=np.float64)
    out = a[::2, ::2]
    mv = meanvar(x, ddof, out=out)
    # For this simple data set, the variance should be
    # calculated exactly, so we can test with assert_equal.
    assert_equal(out, [[1.5, 0.75],
                       [1.5, 1.00],
                       [1.5, 1.50]])
    assert mv is out


def test_length_zero_1d():
    x = np.array([])
    with pytest.raises(ValueError, match="n must be at least 1"):
        meanvar(x, 1)


def test_length_zero_2d():
    x = np.array([[], []])
    with pytest.raises(ValueError, match="n must be at least 1"):
        meanvar(x, 1, axes=[(1,), (), (1,)])


def test_size_zero():
    x = np.zeros((3, 0))
    mv = meanvar(x, 1, axes=[(0,), (), (1,)])
    assert mv.shape == (0, 2)
