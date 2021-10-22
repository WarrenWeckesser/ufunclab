
import numpy as np
from numpy.testing import assert_equal
from ufunclab import issnan


def test_float16():
    x = np.array([1.0, 0, np.inf, -1.0, -np.inf, np.nan, 0], dtype=np.float16)
    v = x.view(np.uint16)
    v[1] = 0b0111_1101_0000_0000
    v[6] = 0b0111_1100_0000_0101
    # This is just to ensure the the binary values put into v are,
    # in fact, nan.
    assert_equal(x, np.array([1, np.nan, np.inf, -1, -np.inf, np.nan, np.nan],
                             dtype=np.float16))
    q = issnan(x)
    assert_equal(q, [False, True, False, False, False, False, True])


def test_float32():
    x = np.array([1.0, 0, np.inf, -1.0, -np.inf, np.nan, 0, 0],
                 dtype=np.float32)
    v = x.view(np.uint32)
    v[1] = 0b0111_1111_1000_0000_0000_0000_0000_0011
    v[6] = 0b0111_1111_1010_0000_0000_0000_0000_0000
    v[7] = 0b0111_1111_1000_0000_0011_0000_0000_0000
    # This is just to ensure the the binary values put into v are,
    # in fact, nan.
    assert_equal(x, np.array([1.0, np.nan, np.inf, -1.0,
                              -np.inf, np.nan, np.nan, np.nan],
                             dtype=np.float32))
    q = issnan(x)
    assert_equal(q, [False, True, False, False, False, False, True, True])


def test_float64():
    x = np.array([1.0, 0, np.inf, -1.0, -np.inf, np.nan, 0, 0, 0],
                 dtype=np.float64)
    v = x.view(np.uint64)
    v[1] = 0x7F_F4_00_00_00_00_00_00
    v[6] = 0x7F_F0_13_00_00_00_00_00
    v[7] = 0x7F_F0_00_03_00_00_00_00
    v[8] = 0x7F_F0_00_00_00_01_20_00
    # This is just to ensure the the floating point values put into v are,
    # in fact, nan.
    assert_equal(x, np.array([1.0, np.nan, np.inf, -1.0,
                              -np.inf, np.nan, np.nan, np.nan, np.nan],
                             dtype=np.float64))
    q = issnan(x)
    assert_equal(q, [False, True, False, False, False, False,
                     True, True, True])
