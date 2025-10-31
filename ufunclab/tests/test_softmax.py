import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ufunclab import softmax


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_x_has_nan(dtype):
    x = np.array([1.0, -10.0, np.nan, 3.0, 999.0], dtype=dtype)
    y = softmax(x)
    assert np.all(np.isnan(y))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_x_has_two_pos_inf(dtype):
    x = np.array([1.0, -10.0, np.inf, 3.0, np.inf, 999.0], dtype=dtype)
    y = softmax(x)
    assert np.all(np.isnan(y))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_x_has_one_pos_inf(dtype):
    x = np.array([1.0, -10.0, np.inf, 3.0, 999.0, -np.inf, -2322.5], dtype=dtype)
    y = softmax(x)
    assert_array_equal(y, np.isposinf(x).astype(dtype), strict=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_x_all_same(dtype):
    x = np.array([10, 10, 10, 10], dtype=dtype)
    y = softmax(x)
    assert_array_equal(y, np.full_like(x, fill_value=1/len(x)), strict=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_x_all_neg_inf(dtype):
    x = np.array([-np.inf, -np.inf, -np.inf], dtype=dtype)
    y = softmax(x)
    assert np.all(np.isnan(y))


def test_basic():
    x = [-3, -2, 1.5, 16, 8, 3, 4]
    y = softmax(x)
    # Reference value computed with mpmath.
    ref = [5.6008675416020074e-09,
           1.5224736461942822e-08,
           5.041740288892188e-07,
           0.9996557262151563,
           0.0003353471369139302,
           2.2595512348211195e-06,
           6.142097062086446e-06]
    assert_allclose(y, ref, rtol=1e-14)
