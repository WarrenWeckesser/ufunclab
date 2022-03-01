import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import fillnan1d


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_all_nan(dtype):
    x = np.full(10, fill_value=np.nan, dtype=dtype)
    y = fillnan1d(x)
    assert_array_equal(y, x)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_basic(dtype):
    x = np.array([dtype(1), dtype('nan'), dtype('nan'), dtype(4), dtype(2)])
    y = fillnan1d(x)
    expected = np.array([dtype(1), dtype(2), dtype(3), dtype(4), dtype(2)])
    assert_array_equal(y, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_basic_with_nan_ends(dtype):
    x = np.array([dtype('nan'), dtype(2), dtype(1), dtype('nan'), dtype('nan'),
                  dtype(4), dtype('nan')])
    y = fillnan1d(x)
    expected = np.array([dtype(2), dtype(2), dtype(1), dtype(2), dtype(3),
                         dtype(4), dtype(4)])
    assert_array_equal(y, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_only_one_not_nan(dtype):
    x = np.array([dtype('nan'), dtype('nan'), dtype('nan'), dtype(4),
                  dtype('nan')])
    y = fillnan1d(x)
    expected = np.array([dtype(4), dtype(4), dtype(4), dtype(4),
                         dtype(4)])
    assert_array_equal(y, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_empty(dtype):
    x = np.array([], dtype=dtype)
    y = fillnan1d(x)
    assert_array_equal(y, x)
