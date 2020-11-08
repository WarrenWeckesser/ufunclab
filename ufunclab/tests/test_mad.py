
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from ufunclab import mad, mad1, rmad, rmad1


@pytest.mark.parametrize('func', [mad, mad1, rmad, rmad1])
def test_nonzero_constant_input(func):
    x = np.array([100, 100, 100, 100], dtype=np.int8)
    m = func(x)
    assert_equal(m, 0)


def test_basic_mad():
    x = [1, 2, 3, 4]
    assert_equal(mad(x), 20/16)


def test_basic_mad1():
    x = [1, 2, 3, 4]
    assert_allclose(mad1(x), 20/12, rtol=1e-15)


def test_basic_rmad():
    x = [1, 2, 3, 4]
    assert_allclose(rmad(x), 0.5, rtol=1e-15)


def test_basic_rmad1():
    x = [1, 2, 3, 4]
    assert_allclose(rmad1(x), 2/3, rtol=1e-15)


@pytest.mark.parametrize('func', [rmad, rmad1])
def test_all_zeros(func):
    x = [0, 0, 0, 0]
    m = func(x)
    assert_equal(m, np.nan)


@pytest.mark.parametrize('func', [rmad, rmad1])
def test_zero_mean(func):
    x = [1, 2, 3, -2, -3, -1]
    m = func(x)
    assert_equal(m, np.inf)
