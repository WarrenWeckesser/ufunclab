import pytest
import math
import numpy as np
from numpy.testing import assert_equal
from ufunclab import logfactorial


@pytest.mark.parametrize('x', [0, 1, 2, [0, 2, 4, 6], 25, 2000])
def test_basic_equal(x):
    if isinstance(x, list):
        expected = [math.log(math.factorial(t)) for t in x]
    else:
        expected = math.log(math.factorial(x))
    y = logfactorial(x)
    assert_equal(y, expected)


def test_bad_input():
    assert_equal(logfactorial(-1), np.nan)
