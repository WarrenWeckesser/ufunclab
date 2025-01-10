
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ufunclab import wjaccard


@pytest.mark.parametrize('dt', [np.int8, np.uint8, np.int16, np.uint16,
                                np.int32, np.uint32, np.int64, np.uint64,
                                np.float64])
def test_wjaccard_basic(dt):
    x = np.array([0, 1, 2, 4, 5], dtype=dt)
    y = np.array([1, 1, 1, 2, 2], dtype=dt)
    w = wjaccard(x, y)
    assert_allclose(w, 6/13, rtol=1e-15)


def test_wjaccard_float32():
    x = np.array([1.0, 2.5, 3.0, 0.5, 0.5], dtype=np.float32)
    y = np.array([2.0, 0.5, 2.5, 0.5, 2.0], dtype=np.float32)
    w = wjaccard(x, y)
    assert_allclose(w, 0.5, rtol=1e-7)


def test_wjaccard_float32_with_nan():
    x = np.array([1.0, 2.5, 3.0, np.nan, 0.5], dtype=np.float32)
    y = np.array([2.0, 0.5, 2.5, 0.5, 2.0], dtype=np.float32)
    w = wjaccard(x, y)
    assert np.isnan(w)
