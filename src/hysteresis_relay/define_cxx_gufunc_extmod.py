

from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


HYSTERESIS_RELAY_DOCSTRING = """\
hysteresis_relay(x, low_threshold, high_threshold, low_value, high_value, init, /, ...)

Pass x through a 'relay' with hysteresis.

Parameters
----------
x : array_like
    Input signal
low_threshold : scalar
    Low end of hysteresis interval.
high_threshold : scalar
    High end of the hysteresis interval.
low_value : scalar
    Output value for x < low_threshold.
high_value : scalar
    Outout value for x > high_threshold.
init : scalar
    Initial output value if the initial value of x is
    between low_threshold and high_threshold.  Normally
    this would be either low_value or high_value, but
    the function does not require it.

Returns
-------
out : ndarray
    Output of the relay.

Notes
-----
The function expects ``low_threshold <= high_threshold``.
It does not check that this condition is satisfied.

Examples
--------
>>> import numpy as np
>>> from ufunclab import hysteresis_relay
>>> x = np.array([-0.2, -0.6, -2, 0.2, 1.2, 2, 0.5, -0.7, -0.2, 0.7])

`x` is the input signal.  The lower and upper thresholds
are -0.5 and 0.5, respectively. The low and high output
values are -1 and 1 (except for the initial output, which
is 0, as determined by the last argument of `hysteresis_relay`).

>>> hysteresis_relay(x, -0.5, 0.5, -1, 1, 0)
array([ 0., -1., -1., -1.,  1.,  1.,  1., -1., -1.,  1.])
"""


hysteresis_relay_core_source = UFuncSource(
    funcname='hysteresis_relay_core',
    typesignatures=['ffffff->f', 'dddddd->d', 'gggggg->g'],
)

hysteresis_relay_gufunc = UFunc(
    name='hysteresis_relay',
    docstring=HYSTERESIS_RELAY_DOCSTRING,
    header='hysteresis_relay_gufunc.h',
    signature='(n),(),(),(),(),() -> (n)',
    sources=[hysteresis_relay_core_source],
)


MODULE_DOCSTRING = """\
This module defines the hysteresis_relay function.
"""

extmod = UFuncExtMod(
    module='_hysteresis_relay',
    docstring=MODULE_DOCSTRING,
    ufuncs=[hysteresis_relay_gufunc],
)
