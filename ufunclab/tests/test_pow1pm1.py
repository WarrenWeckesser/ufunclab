import pytest
from numpy.testing import assert_allclose
from ufunclab import pow1pm1


@pytest.mark.parametrize('x, y, expected', [
    (-0.125, 0.125, -0.016552894084366807),
    (2e-18, 4.25, 8.500000000000001e-18),
])
def test_logexpint1(x, y, expected):
    z = pow1pm1(x, y)
    assert_allclose(z, expected, rtol=1e-15)
