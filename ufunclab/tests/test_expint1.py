import numpy as np
from numpy.testing import assert_allclose
from ufunclab import expint1, logexpint1


def test_expint1_basic():
    x = np.array([0.1, 1, 2.5, 25, 450, 600])
    y = expint1(x)
    # The expected values were computed with mpmath.
    expected = [1.8229239584193906,
                0.21938393439552029,
                0.024914917870269736,
                5.348899755340217e-13,
                8.190468180358912e-199,
                4.409989794509838e-264]
    assert_allclose(y, expected, rtol=2e-15)


def test_logexpint1_basic():
    x = np.array([0.1, 1, 2.5, 25, 450, 600])
    y = logexpint1(x)
    # The expected values were computed with mpmath.
    expected = [0.6004417824948862,
                -1.5169319590020456,
                -3.6922885436511623,
                -28.256715322371637,
                -456.11146244470496,
                -606.3985921751421]
    assert_allclose(y, expected, rtol=2e-15)
