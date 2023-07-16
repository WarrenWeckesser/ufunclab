import pytest
from numpy.testing import assert_allclose
from ufunclab import loggamma1p


# Expected values were computed with mpmath.loggamma(mp.one + x)
# with mp.dps = 100.
@pytest.mark.parametrize(
    'x, expected',
    [(-0.975, 3.674956947385164),
     (-0.125, 0.08585870722533433),
     (-1e-8, 5.772156731262032e-09),
     (-3e-75, 1.7316469947045985e-75),
     (2e-75, -1.1544313298030657e-75),
     (6e-57, -3.463293989409197e-57),
     (3e-8, -1.7316469206825664e-08),
     (0.0625, -0.03295710029357782),
     (0.75, -0.08440112102048555),
     (1.5, 0.2846828704729192),
     (2.0, 0.6931471805599453),
     (2.5, 1.2009736023470743)]
)
def test_basic(x, expected):
    y = loggamma1p(x)
    assert_allclose(y, expected)


def test_zeros():
    assert loggamma1p(0) == 0.0
    assert loggamma1p(1) == 0.0
