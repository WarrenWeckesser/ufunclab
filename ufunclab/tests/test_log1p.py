import pytest
from numpy.testing import assert_allclose
from ufunclab import log1p


# Reference values were computed with mpmath.
@pytest.mark.parametrize(
    'z, wref',
    [(0.0, 0.0 + 0.0j),
     (1.0, 0.6931471805599453 + 0j),
     (-0.57113 - 0.90337j, 3.4168883248419116e-06 - 1.1275564209486122j),
     (-0.978249978133276 - 0.015379941691430407j,
      -3.6254002951579887 - 0.6154905511361795j),
     (1e200 + 1e200j, 460.8635921890891 + 0.7853981633974483j)],
)
def test_basic(z, wref):
    w = log1p(z)
    assert_allclose(w.real, wref.real, rtol=1e-15)
    assert_allclose(w.imag, wref.imag, rtol=1e-15)


# In this test, the inputs are so small that the output
# should equal the input.
@pytest.mark.parametrize('z', [3e-180+2e-175j, 1e-50-3e-55])
def test_tiny(z):
    w = log1p(z)
    assert w == z
