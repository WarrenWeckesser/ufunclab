import pytest
import numpy as np
from numpy.testing import assert_allclose
from ufunclab import normal, erfcx


f64eps = np.finfo(np.float64).eps
longdouble_eps = np.finfo(np.longdouble).eps

# The expected values were computed with mpmath.


@pytest.mark.parametrize('x, expected, rtol', [
    (-10,    7.619853024160525e-24, 32*f64eps),
    (-0.125, 0.4502617751698871,    f64eps),
    ( 0.0,   0.5,                   f64eps),
    ( 0.125, 0.5497382248301129,    f64eps),
    ( 7.5,   0.9999999999999681,    f64eps),
    ( 10.0,  1.0,                   f64eps),
])
def test_normal_cdf_double(x, expected, rtol):
    y = normal.cdf(x)
    assert_allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize('x, expected, rtol', [
    (-10,    '7.619853024160526066e-24', 32*longdouble_eps),
    (-0.125, '0.450261775169887107',     longdouble_eps),
    ( 0.0,   '0.5',                      longdouble_eps),
    ( 0.125, '0.54973822483011289296',   longdouble_eps),
    ( 7.5,   '0.99999999999996809106',   longdouble_eps),
    ( 10.0,  '1.0',                      longdouble_eps),
])
def test_normal_cdf_longdouble(x, expected, rtol):
    if np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    x = np.longdouble(x)
    expected = np.longdouble(expected)
    y = normal.cdf(x)
    assert_allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize('x, expected, rtol', [
    (-100,   -5005.524208694205,     f64eps),
    (-10,    -53.23128515051247,     f64eps),
    (-0.125, -0.7979261427530241,    1.5*f64eps),
    ( 0.0,   -0.6931471805599453,    f64eps),
    ( 0.125, -0.598313068912425,     f64eps),
    ( 7.5,   -3.190891672910947e-14, 16*f64eps),
    ( 10.0,  -7.619853024160525e-24, 32*f64eps),
])
def test_normal_logcdf_double(x, expected, rtol):
    y = normal.logcdf(x)
    assert_allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize('x, expected, rtol', [
    (-100,   '-5005.52420869420508862630245733002553', longdouble_eps),
    (-10,    '-53.2312851505124705783470273541312099', longdouble_eps),
    (-0.125, '-0.797926142753024090241059376857968502', longdouble_eps),
    ( 0.0,   '-0.693147180559945309417232121458176568', longdouble_eps),
    ( 0.125, '-0.598313068912424993857612822292078551', longdouble_eps),
    ( 7.5,   '-3.19089167291094713671562960629911811e-14', 32*longdouble_eps),
    ( 10.0,  '-7.61985302416052606597337228267936327e-24', 32*longdouble_eps),
])
def test_normal_logcdf_longdouble(x, expected, rtol):
    if np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    x = np.longdouble(x)
    expected = np.longdouble(expected)
    y = normal.logcdf(x)
    assert_allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize('x, expected, rtol', [
    (-25.0,   5.4335189393274735e+271, f64eps),
    (-10.0,   5.376234283632271e+43,   f64eps),
    ( 0.0,    1.0,                     f64eps),
    ( 10.0,   0.05614099274382259,     f64eps),
    ( 250.0,  0.002256740280557632,    f64eps),
    ( 2500.0, 0.00022567581536504018,  f64eps),
    ( 1e8,    5.641895835477562e-09,   f64eps),
])
def test_erfcx_double(x, expected, rtol):
    y = erfcx(x)
    assert_allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize('x, expected, rtol', [
    (-40.0,   '1.48662366149239608757079824785822559e+695', longdouble_eps),
    (-25.0,   '5.43351893932747338681438935625175482e+271', longdouble_eps),
    (-10.0,   '5.37623428363227089682525110316002717e+43',  longdouble_eps),
    ( 0.0,    '1.0',                                        longdouble_eps),
    ( 10.0,   '0.05614099274382258585751738722046831157',   longdouble_eps),
    ( 250.0,  '0.002256740280557631888822322457284549275',  longdouble_eps),
    ( 2500.0, '0.0002256758153650401742252990556239775639', longdouble_eps),
    ( 1e8,    '5.641895835477562587386002741729624699e-9',  longdouble_eps),
])
def test_erfcx_longdouble(x, expected, rtol):
    if np.dtype('g') == np.dtype('d'):
        pytest.skip('longdouble is double')
    x = np.longdouble(x)
    expected = np.longdouble(expected)
    y = erfcx(x)
    assert_allclose(y, expected, rtol=rtol)
