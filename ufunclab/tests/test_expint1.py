import pytest
import numpy as np
from numpy.testing import assert_allclose
from ufunclab import expint1, logexpint1


ldeps = np.finfo(np.longdouble).eps


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


@pytest.mark.parametrize('x, expected, rtol', [
    ('0.00390625', '4.97186421818970933971749385107140230',       ldeps),
    ('3',          '0.0130483810941970374125007458286450229',     16*ldeps),
    ('498',        '1.05499209080431611958074009093339023e-219',  ldeps),
    ('500.25',     '1.10696213091972157838312751368960077e-220',  ldeps),
    ('2500.0',     '7.33975594722389377705530492879166952e-1090', 2*ldeps)
])
def test_expint1_longdouble(x, expected, rtol):
    x = np.longdouble(x)
    expected = np.longdouble(expected)
    y = expint1(x)
    assert_allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize('x, expected, rtol', [
    ('2.5e-800', '7.51783306707495062165187327674702238',   ldeps),
    ('0.25',     '0.0433301754689423938905717824826649295', 16*ldeps),
    ('0.50',     '-0.580222872044787464047609353147219066', 2*ldeps),
    ('2500.0',   '-2507.82444577113317199235247002342904',  ldeps),
])
def test_logexpint1_longdouble(x, expected, rtol):
    x = np.longdouble(x)
    expected = np.longdouble(expected)
    y = logexpint1(x)
    assert_allclose(y, expected, rtol=rtol)
