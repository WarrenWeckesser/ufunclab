
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from ufunclab import mad, rmad, gini


@pytest.mark.parametrize('func', [mad, rmad])
@pytest.mark.parametrize('unbiased', [False, True])
def test_nonzero_constant_input(func, unbiased):
    x = np.array([100, 100, 100, 100], dtype=np.int8)
    m = func(x, unbiased)
    assert_equal(m, 0)


@pytest.mark.parametrize('unbiased, expected', [(False, 20/16), (True, 20/12)])
def test_basic_mad(unbiased, expected):
    x = [1, 2, 3, 4]
    assert_equal(mad(x, unbiased), expected)


@pytest.mark.parametrize('unbiased', [False, True])
def test_basic_mad_bigger(unbiased):
    x = [1, 3, 6, 10, 11, 12, 12, 13, 13, 15, 12, 4, 4, 4, 5, 5]
    s = np.abs(np.subtract.outer(x, x)).sum()
    n = len(x)
    if unbiased:
        expected = s / (n*(n - 1))
    else:
        expected = s / n**2
    assert_allclose(mad(x, unbiased), expected, rtol=1e-15)


@pytest.mark.parametrize('unbiased, expected', [(False, 0.5), (True, 2/3)])
def test_basic_rmad(unbiased, expected):
    x = [1, 2, 3, 4]
    assert_allclose(rmad(x, unbiased), expected, rtol=1e-15)


@pytest.mark.parametrize('func', [rmad, gini])
@pytest.mark.parametrize('unbiased', [False, True])
def test_rmad_gini_all_zeros(func, unbiased):
    x = [0, 0, 0, 0]
    m = func(x, unbiased)
    assert_equal(m, np.nan)


@pytest.mark.parametrize('unbiased', [False, True])
def test_rmad_zero_mean(unbiased):
    x = [1, 2, 3, -2, -3, -1]
    m = rmad(x, unbiased)
    assert_equal(m, np.inf)


@pytest.mark.parametrize('unbiased, expected', [(False, 0.25), (True, 1/3)])
def test_basic_gini(unbiased, expected):
    x = [1, 2, 3, 4]
    g = gini(x, unbiased)
    assert_allclose(g, expected, rtol=1e-15)
