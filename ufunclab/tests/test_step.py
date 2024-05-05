from numpy.testing import assert_equal
from ufunclab import step, linearstep


def test_step_basic():
    a = 2.0
    flow = -1.0
    fa = 1.0
    fhigh = 5.0
    y = step([a - 3, a - 0.25, a, a + 1, a + 10], a, flow, fa, fhigh)
    assert_equal(y, [flow, flow, fa, fhigh, fhigh])


def test_linearstep_basic():
    a = 2.0
    fa = 1.0
    b = 4.0
    fb = 9.0
    y = linearstep([a - 3, a, (a + b)/2, b, b + 10], a, b, fa, fb)
    # With the values of the parameters used in this test, and the
    # specific interior point at which the y value is tests, `assert_equal`
    # should be safe to use (instead of `assert_allclose`).
    assert_equal(y, [fa, fa, (fa + fb)/2, fb, fb])
