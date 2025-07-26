
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from ufunclab import gmean, hmean, pmean


@pytest.mark.parametrize('func', [gmean, hmean])
def test_constant_input(func):
    x = np.array([100, 100, 100, 100], dtype=np.int8)
    m = func(x)
    assert_allclose(m, x[0], rtol=1e-14)


@pytest.mark.parametrize('dt, rtol', [(np.float32, 1e-7), (np.float64, 1e-14)])
def test_gmean_float(dt, rtol):
    x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0], dtype=dt)
    m = gmean(x)
    assert_equal(m.dtype, dt)
    assert_allclose(m, 100.0, rtol=rtol)


def test_gmean_with_zero_or_neg():
    x = np.array([[1.0, 9.0, 0.0, 4.0],
                  [2.0, 2.0, 3.0, -3.0],
                  [3.0, 0.0, 3.0, -3.0],
                  [3.0, np.inf, 3.0, 0.0]])
    m = gmean(x)
    assert_equal(m, [0.0, np.nan, np.nan, np.nan])


def test_gmean_with_nan():
    x = np.array([[3.0, 4.0, np.nan, 5.5],
                  [3.0, 0.0, np.nan, 9.0]])
    m = gmean(x, axis=1)
    assert_equal(m, [np.nan, np.nan])


def test_gmean_with_inf():
    x = np.array([[3.0, 4.0, np.inf, 5.5],
                  [3.0, np.inf, np.inf, 9.0]])
    m = gmean(x, axis=1)
    assert_equal(m, [np.inf, np.inf])


def test_gmean_axis():
    x = np.array([[1, 2, 4, 8, 16],
                  [3, 3, 3, 3, 3],
                  [1, 2, 3, 4, 5]])
    m = gmean(x, axis=1)
    assert_allclose(m, [4.0, 3.0, 120**0.2], rtol=1e-14)


def test_gmean_empty_array():
    with pytest.raises(ValueError, match='length at least 1'):
        gmean([])


@pytest.mark.parametrize('dt, rtol', [(np.float32, 1e-7), (np.float64, 1e-14)])
def test_hmean_float(dt, rtol):
    x = np.array([1.0, 1.0, 4.0, 4.0, 4.0, 4.0], dtype=dt)
    m = hmean(x)
    assert_equal(m.dtype, dt)
    assert_allclose(m, 2.0, rtol=rtol)


def test_hmean_axis():
    x = np.array([[16, 32, 32, 32, 32, 16],
                  [11, 11, 11, 11, 11, 11],
                  [2, 4, 1, 1, 1, 4]])
    m = hmean(x, axis=1)
    assert_allclose(m, [24.0, 11.0, 1.5], rtol=1e-14)


def test_hmean_empty_array():
    with pytest.raises(ValueError, match='length at least 1'):
        hmean([])


def test_pmean_basic():
    x = np.array([1, 2, 3, 4, 5])
    p = [2.0, 2.5, 3.0]
    m = pmean(x, p)
    # Reference values were computed with mpsci.stats.pmean.
    ref = [3.3166247903554, 3.4450940679055617, 3.556893304490063]
    assert_allclose(m, ref, rtol=1e-15)


def test_pmean_constant_input():
    x = np.full(8, fill_value=100.0)
    m = pmean(x, 3.0)
    assert_equal(m, x[0])


def test_pmean_big_values():
    x = 1e155*np.arange(10)
    y = pmean(x, 3)
    # Reference value was computed with mpsci.stats.pmean.
    assert_allclose(y, 5.872301461753295e+155, rtol=1e-15)


def test_pmean_small_values():
    x = 1e-155*np.arange(10)
    y = pmean(x, 3)
    # Reference value was computed with mpsci.stats.pmean.
    assert_allclose(y, 5.872301461753296e-155, rtol=1e-15)
