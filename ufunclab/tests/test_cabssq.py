
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ufunclab import cabssq


@pytest.mark.parametrize('typ', [np.complex64, np.complex128])
def test_basic(typ):
    z = np.array([3+4j, 1-1j, 0, -1.5, 2.5j], dtype=typ)
    w = cabssq(z)
    realtype = typ(0).real.dtype
    assert w.dtype == realtype
    assert_array_equal(w, [25.0, 2.0, 0.0, 2.25, 6.25])
