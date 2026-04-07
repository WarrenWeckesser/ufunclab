import pytest
import numpy as np
from ufunclab import kbnsum


def test_kbnsum_case1():
    # Example from https://github.com/numpy/numpy/issues/8786
    p = np.array([
        -0.41253261766461263,
        41287272281118.43,
        -1.4727977348624173e-14,
        5670.3302557520055,
        2.119245229045646e-11,
        -0.003679264134906428,
        -6.892634568678797e-14,
        -0.0006984744181630712,
        -4054136.048352595,
        -1003.101760720037,
        -1.4436349910427172e-17,
        -41287268231649.57,
    ])
    sum = kbnsum(p)
    # Reference value computed with mpmath.
    ref = -0.377392919181026
    # Asserting *equality* might be too optimistic, but let's give it a shot.
    assert sum == ref


def test_kbnsum_case2():
    x = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1e-9, 0.0, -1.5, 0.0, 0.5, -1.0, 0.0])
    sum = kbnsum(x)
    # Reference value computed with mpmath.
    ref = 1e-9
    # Asserting *equality* might be too optimistic, but let's give it a shot.
    assert sum == ref


def test_inf():
    x = [1.24, -np.inf, 7.0, -np.inf, -312.5, -np.inf, 3319.0]
    sum = kbnsum(x)
    assert sum == -np.inf


def test_nan():
    x = [-3.5, 10.0, 0.125, np.nan, 3.0, np.nan]
    sum = kbnsum(x)
    assert np.isnan(sum)


@pytest.mark.filterwarnings('ignore:invalid value')
def test_mixed_sign_infs():
    x = [1.24, np.inf, 7.0, np.inf, -312.5, -np.inf, 3319.0]
    sum = kbnsum(x)
    assert np.isnan(sum)

