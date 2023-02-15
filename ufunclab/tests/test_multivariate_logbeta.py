
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ufunclab import multivariate_logbeta


def test_length_zero():
    y = multivariate_logbeta([])
    assert np.isnan(y)


def test_length_one():
    y = multivariate_logbeta([2.5])
    assert y == 0.0


# Expected values that are not a formula were computed with
# mpsci.fun.multivariate_logbeta.
@pytest.mark.parametrize('x, expected',
                         [([1, 1, 1], -np.log(2)),
                          ([1, 2, 3], -4.0943445622221),
                          ([0.125, 5, 2.5, 12.0, 6.5], -29.761728907094028),
                          ([[1000, 2500, 1250, 800], -7127.642862517964])])
def test_basic(x, expected):
    y = multivariate_logbeta(x)
    assert_allclose(y, expected, rtol=5e-15)
