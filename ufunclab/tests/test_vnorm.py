import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ufunclab import vnorm


def test_all_zeros():
    x = np.zeros(10)
    nrm = vnorm(x, 2)
    assert_array_equal(nrm, 0.0)


@pytest.mark.parametrize('x, order, nrm', [([3, 4], 2, 5),
                                           ([1, 1, 4], 0.5, 16)])
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


def test_order_inf():
    x = np.array([-4, 2, -3.5, 0, 1])
    nrm = vnorm(x, np.inf)
    assert_array_equal(nrm, 4.0)
