import pytest
import numpy as np
from numpy.testing import assert_allclose
from ufunclab import rms


@pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_all_zeros(dtype):
    x = np.zeros(10, dtype=dtype)
    m = rms(x)
    assert m == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_basic_real(dtype):
    x = np.array([4, 1, 4, 4], dtype=dtype)
    m = rms(x)
    assert m.dtype == dtype
    assert_allclose(m, 3.5, rtol=np.finfo(dtype).eps*64)


@pytest.mark.parametrize('dtype, result_type',
                         [(np.complex64, np.float32),
                          (np.complex128, np.float64)])
def test_basic_complex(dtype, result_type):
    z = np.array([4+2j, 4j, 4+4j, 4+4j], dtype=dtype)
    m = rms(z)
    assert m.dtype == result_type
    assert_allclose(m, 5, rtol=np.finfo(dtype).eps*64)
