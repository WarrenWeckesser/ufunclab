import pytest
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from ufunclab import all_same


@pytest.mark.parametrize('typ', [np.int8, np.uint8,
                                 np.int16, np.uint16,
                                 np.int32, np.uint32,
                                 np.int64, np.uint64,
                                 np.float32, np.float64])
def test_basic(typ):
    x = np.ones(8, dtype=typ)
    assert_equal(all_same(x), True)

    x[-2] = 13
    assert_equal(all_same(x), False)

    x = np.array([[1, 2, 3],
                  [1, 1, 1],
                  [1, 1, 2]], dtype=typ)
    assert_array_equal(all_same(x, axis=0), np.array([True, False, False]))
    assert_array_equal(all_same(x, axis=1), np.array([False, True, False]))


@pytest.mark.parametrize('last,result', [(Fraction(1, 3), False),
                                         (Fraction(3, 5), True)])
def test_fraction(last, result):
    x = np.array([Fraction(3, 5), Fraction(3, 5), last])
    assert_equal(all_same(x), result)
