import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from ufunclab import logistic, logistic_deriv, log_logistic


ldeps = np.finfo(np.longdouble).eps


def test_logistic_basic():
    x = np.array([-100.0, -2.5, 0.0, 2.5])
    y = logistic(x)
    # Expected values computed with mpmath.
    expected = [3.720075976020836e-44, 0.0758581800212435, 0.5,
                0.92414181997875645]
    assert_allclose(y, expected, rtol=1e-15)


def test_logistic_deriv_basic():
    x = np.array([-100.0, -2.5, 0.0, 2.5, 100.0])
    y = logistic_deriv(x)
    # Expected values computed with mpmath.
    expected = [3.720075976020836e-44, 0.07010371654510816, 0.25,
                0.07010371654510816, 3.720075976020836e-44]
    assert_allclose(y, expected, rtol=1e-15)


def test_log_logistic_large_negative():
    x = np.array([-10000.0, -750.0, -500.0, -35.0])
    y = log_logistic(x)
    assert_equal(y, x)


def test_log_logistic_large_positive():
    x = np.array([750.0, 1000.0, 10000.0])
    y = log_logistic(x)
    # y will contain -0.0, and -0.0 is used in the expected value,
    # but assert_equal does not check the sign of zeros, and I don't
    # think the sign is an essential part of the test (i.e. it would
    # probably be OK if log_expit(1000) returned 0.0 instead of -0.0).
    assert_equal(y, np.array([-0.0, -0.0, -0.0]))


def test_log_logistic_basic():
    x = np.array([-32, -20, -10, -3, -1, -0.1, -1e-9,
                  0, 1e-9, 0.1, 1, 10, 100, 500, 710, 725, 735])
    y = log_logistic(x)
    #
    # Expected values were computed with mpmath:
    #
    #   import mpmath
    #
    #   mpmath.mp.dps = 100
    #
    #   def mp_log_expit(x):
    #       return -mpmath.log1p(mpmath.exp(-x))
    #
    #   expected = [float(mp_log_expit(t)) for t in x]
    #
    expected = [-32.000000000000014, -20.000000002061153,
                -10.000045398899218, -3.048587351573742,
                -1.3132616875182228, -0.7443966600735709,
                -0.6931471810599453, -0.6931471805599453,
                -0.6931471800599454, -0.6443966600735709,
                -0.3132616875182228, -4.539889921686465e-05,
                -3.720075976020836e-44, -7.124576406741286e-218,
                -4.47628622567513e-309, -1.36930634e-315,
                -6.217e-320]

    # When tested locally, only one value in y was not exactly equal to
    # expected.  That was for x=1, and the y value differed from the
    # expected by 1 ULP.  For this test, however, I'll use rtol=1e-15.
    assert_allclose(y, expected, rtol=1e-15)


@pytest.mark.parametrize('x, expected, rtol', [
    ('-1000', '-1000', ldeps),
    ('-1.0', '-1.313261687518222834048995494967855642', ldeps),
    ('0.0', '-0.693147180559945309417232121458176568', ldeps),
    ('1000', '-5.07595889754945676529180947957433692e-435', ldeps),
])
def test_log_logistic_longdouble(x, expected, rtol):
    x = np.longdouble(x)
    expected = np.longdouble(expected)
    y = log_logistic(x)
    assert_allclose(y, expected, rtol=rtol)
