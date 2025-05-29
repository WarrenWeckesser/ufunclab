import pytest
import numpy as np
from numpy.testing import assert_equal
from ufunclab import nan_to_num


@pytest.mark.parametrize('typ', [np.float64, np.float32])
def test_nan_to_num_basic(typ):
    x = np.array([-100.0, np.nan, 2.5, np.nan], dtype=typ)
    replacement = typ(999.25)
    y = nan_to_num(x, replacement)
    expected = x.copy()
    expected[np.isnan(expected)] = replacement
    assert_equal(y, expected)

    nan_to_num(x, replacement, out=x)
    assert_equal(x, expected)
