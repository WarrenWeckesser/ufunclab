import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ufunclab import vnorm


def test_all_zeros():
    x = np.zeros(10)
    nrm = vnorm(x, 2)
    assert_array_equal(nrm, 0.0)


def test_all_zeros_complex():
    z = np.zeros(10, dtype=np.complex128)
    nrm = vnorm(z, 2)
    assert_array_equal(nrm, 0.0)


@pytest.mark.parametrize('order', [0.25, 1, 2, 2.5, 11, np.inf])
@pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_empty(order, dtype):
    x = np.array([], dtype=dtype)
    assert vnorm(x, order) == 0.0


@pytest.mark.parametrize('x, order, nrm', [([3, 4], 2, 5),
                                           ([1, 1, 4], 0.5, 16),
                                           ([3+4j, 0, 3], 1, 8),
                                           ([3+4j, 0, 3], np.inf, 5),
                                           ([4-2j, 4, 4, 12], 2, 14)])
def test_basic(x, order, nrm):
    result = vnorm(x, order)
    assert_allclose(result, nrm, rtol=1e-13)


def test_broadcasting():
    x = np.array([[0, 2, 3],
                  [6, 5, 4]])
    p = np.array([[1], [2]])
    nrm = vnorm(x, p)
    assert_allclose(nrm, [[5, 15], [np.sqrt(13), np.sqrt(77)]])

    nrm0 = vnorm(x, p, axis=0)
    assert_allclose(nrm0, [[6, 7, 7], [6, np.sqrt(29), 5]])


def test_big_values_order2():
    x = np.array([3e200, 4e200])
    nrm = vnorm(x, 2)
    assert_allclose(nrm, 5e200, rtol=1e-13)


def test_big_values_order3():
    x = np.full(8, fill_value=1e125)
    nrm = vnorm(x, 3)
    assert_allclose(nrm, 2e125, rtol=1e-13)


@pytest.mark.parametrize('order, expected', [(1, 20e200),
                                             (2, 10e200),
                                             (3, np.cbrt(4)*5e200),
                                             (np.inf, 5e200)])
def test_big_complex(order, expected):
    z = np.array([3e200+4e200j, 4e200-3e200j, -3e200+4e200j, -4e200-3e200j])
    nrm = vnorm(z, order)
    assert_allclose(nrm, expected)


def test_order_inf():
    x = np.array([-4, 2, -3.5, 0, 1])
    nrm = vnorm(x, np.inf)
    assert_array_equal(nrm, 4.0)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_axis_a(dtype):
    a = np.array([[1, 0, 3],
                  [0, 2, 4]], dtype=dtype)
    nrm = vnorm(a, 2, axis=0)
    assert_allclose(nrm, [1, 2, 5])


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_axis_z(dtype):
    z = np.array([[2j, 3+4j, -0j, 14],
                  [0j, 1-2j, -3j, 2-2j]], dtype=dtype)
    nrm = vnorm(z, 2, axis=0)
    assert_allclose(nrm, [2, np.sqrt(30), 3, np.sqrt(204)])


def test_nontrivial_strided():
    a = np.zeros((3, 8))
    a[:,6:] = 10
    b = a[::2, ::2]
    n1 = vnorm(b, 2, axis=1)
    assert_array_equal(n1, [10.0, 10.0])
