
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import abssq


@pytest.mark.parametrize('typ', [np.float32, np.float64])
def test_basic(typ):
    x = np.array([5, -1, 0, -1.5, 2.5j], dtype=typ)
    y = abssq(x)
    assert y.dtype == typ
    assert_array_equal(w, [25.0, 1.0, 0.0, 2.25, 6.25])


@pytest.mark.parametrize('typ', [np.complex64, np.complex128])
def test_basic(typ):
    z = np.array([3+4j, 1-1j, 0, -1.5, 2.5j], dtype=typ)
    w = abssq(z)
    realtype = typ(0).real.dtype
    assert w.dtype == realtype
    assert_array_equal(w, [25.0, 2.0, 0.0, 2.25, 6.25])