import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ufunclab import tri_area


@pytest.mark.parametrize('dt', [np.float32, np.float64, np.longdouble])
def test_all_zeros(dt):
    if dt == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    x = np.zeros((5, 3, 4), dtype=dt)
    a = tri_area(x)
    assert_array_equal(a, np.zeros(5))


@pytest.mark.parametrize('dt', [np.float32, np.float64, np.longdouble])
def test_basic_2d(dt):
    if dt == np.longdouble and np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    p = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 3.0]], dtype=dt)
    a = tri_area(p)
    assert_allclose(a, 1.0, rtol=64*np.finfo(dt).eps)


def test_basic_n2d():
    p = np.array([[[1.0, 1.0], [2.0, 1.0], [2.0, 3.0]],
                  [[0.0, 0.0], [4.0, 4.0], [0.0, 8.0]],
                  [[0.1, 0.1], [4.1, 4.1], [0.1, 8.1]],
                  [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]])
    a = tri_area(p)
    assert_allclose(a, [1.0, 16.0, 16.0, 0.0], rtol=1e-14)

    a = tri_area(p[:, :, ::-1])
    assert_allclose(a, [1.0, 16.0, 16.0, 0.0], rtol=1e-14)

    a = tri_area(p[:, ::-1, :])
    assert_allclose(a, [1.0, 16.0, 16.0, 0.0], rtol=1e-14)

    a = tri_area(p[::-1, ::-1, ::-1])
    assert_allclose(a, [0.0, 16.0, 16.0, 1.0], rtol=1e-14)


def test_3d():
    # Use the cross product to compute the area and compare to
    # the result of tri_area(p).
    n = 4
    p = np.sin(np.arange(n*9)).reshape((n, 3, 3))
    v1 = p[:, 1, :] - p[:, 0, :]
    v2 = p[:, 2, :] - p[:, 0, :]
    ac = np.linalg.norm(np.cross(v1, v2), axis=1)/2
    a = tri_area(p)
    assert_allclose(a, ac, 1e-14)


def test_4d():
    p = np.array([[1.0, 2.0, 3.0, 0.0],
                  [0.0, 2.0, 4.0, 0.0],
                  [1.0, 2.0, 5.0, 0.0]])
    a = tri_area(p)
    assert_allclose(a, 1.0, rtol=1e-14)
