import pytest
from numpy.testing import assert_allclose
from ufunclab import log1p_doubledouble


# Reference values were computed with mpmath.
@pytest.mark.parametrize(
    'z, wref',
    [(0.0, 0.0 + 0.0j),
     (1.0, 0.6931471805599453 + 0j),
     (-0.57113 - 0.90337j, 3.4168883248419116e-06 - 1.1275564209486122j),
     (-0.978249978133276 - 0.015379941691430407j,
      -3.6254002951579887 - 0.6154905511361795j),
     (-0.011228922063957758 + 0.14943813247359922j,
      -4.499779370610757e-17 + 0.15j),
     (-1.9999999995065196 - 3.141592653092555e-05j,
      -1.4766035946815242e-16 - 3.1415612376632573j),
     (-1.25e-8 + 5e-12j, -1.2500000078124988e-08 + 5.0000000625e-12j),
     (1e200 + 1e200j, 460.8635921890891 + 0.7853981633974483j)],
)
def test_basic(z, wref):
    w = log1p_doubledouble(z)
    # Note that we test the relative error of the real and imaginary parts
    # separately.  This is a more stringent test than abs(w - wref)/abs(wref).
    assert_allclose(w.real, wref.real, rtol=1e-15)
    assert_allclose(w.imag, wref.imag, rtol=1e-15)


# In this test, the inputs are so small that the output
# should equal the input.
@pytest.mark.parametrize('z', [3e-180+2e-175j, 1e-50-3e-55j])
def test_tiny(z):
    w = log1p_doubledouble(z)
    assert w == z


# These are values where real part is not computed with
# extremely high precision.
@pytest.mark.parametrize(
    'z, wref',
    [(-1.25e-05 + 0.005j, 7.812499999381789e-11 + 0.005000020833177082j),
     (-8e-8 - 0.0004j, 3.2000000031864308e-15 - 0.00040000001066666614j),
     (-0.01524813-0.173952j, -2.228118716056777e-06-0.1748418364650139j)],
)
def test_real_part_loosely(z, wref):
    w = log1p_doubledouble(z)
    # The computed value is very close to the reference value...
    assert_allclose(w, wref, rtol=1e-15)
    # ...but the real parts (which are very small) have greater relative
    # error...
    assert_allclose(w.real, wref.real, rtol=5e-9)
