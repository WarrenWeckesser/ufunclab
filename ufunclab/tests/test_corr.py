
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ufunclab import pearson_corr


def test_pearson_corr_basic():
    x = np.array([-1, 0, 1])
    y = np.array([0, 0, 3])
    r = pearson_corr(x, y)
    assert_allclose(r, np.sqrt(3)/2, rtol=5e-15)


@pytest.mark.parametrize('sign1, sign2, expected', [(1, 1, 1),
                                                    (1, -1, -1),
                                                    (-1, 1, -1),
                                                    (-1, -1, 1)])
def test_pearson_corr_length2(sign1, sign2, expected):
    x = sign1*np.array([1, 2])
    y = sign2*np.array([3, 5.6])
    r = pearson_corr(x, y)
    assert r == expected


def test_pearson_corr_trivial():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([8.0, 7.0, 6.0])
    r = pearson_corr(x, y)
    assert_allclose(r, -1, rtol=5e-15)


def test_pearson_corr_very_small_input_values():
    # This test is from the SciPy unit tests.
    # Very small values in an input.  A naive implementation will
    # suffer from underflow.
    # See https://github.com/scipy/scipy/issues/9353
    x = [0.004434375, 0.004756007, 0.003911996, 0.0038005, 0.003409971]
    y = [2.48e-188, 7.41e-181, 4.09e-208, 2.08e-223, 2.66e-245]
    r = pearson_corr(x, y)

    # The expected values were computed using mpmath with 80 digits
    # of precision.
    assert_allclose(r, 0.7272930540750450, rtol=5e-15)


def test_extremely_large_input_values():
    # This test is from the SciPy unit tests.
    # Extremely large values in x and y.  These values would cause the
    # product sigma_x * sigma_y to overflow if the two factors were
    # computed independently.
    x = np.array([2.3e200, 4.5e200, 6.7e200, 8e200])
    y = np.array([1.2e199, 5.5e200, 3.3e201, 1.0e200])
    r = pearson_corr(x, y)

    # The expected values were computed using mpmath with 80 digits
    # of precision.
    assert_allclose(r, 0.351312332103289, 5e-15)
