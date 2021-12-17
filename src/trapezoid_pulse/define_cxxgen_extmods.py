
from ufunc_config_types import ExtMod, Func



trapezoid_pulse_docstring = """\
trapezoid_pulse(x, a, b, c, d, amp, /, ...)

Trapezoid pulse function.

Parameters
----------
x : array_like
    Input signal
a : array_like
    Start of rising ramp
b : array_like
    End of rising ramp, start of plateau
c : array_like
    End of plateau, start of falling ramp
d : array_like
    End of falling ramp
amp : array_like
    Height of the plateau

Returns
-------
out : ndarray
    Output of the trapezoid pulse function.

Notes
-----
The function requires ``a <= b <= c <= d``.  If this condition
is not satisfied, nan is returned.

Examples
--------
>>> import numpy as np
>>> from ufunclab import trapezoid_pulse
>>> x = np.linspace(0, 6, 17)
>>> trapezoid_pulse(x, 1, 3, 4, 5, 2)
array([0.   , 0.   , 0.   , 0.125, 0.5  , 0.875, 1.25 , 1.625, 2.   ,
       2.   , 2.   , 1.75 , 1.   , 0.25 , 0.   , 0.   , 0.   ])
"""


funcs = [
    Func(cxxname='trapezoid_pulse',
         ufuncname='trapezoid_pulse',
         types=['ffffff->f', 'dddddd->d', 'gggggg->g'],
         docstring=trapezoid_pulse_docstring),
]

extmods = [ExtMod(modulename='_trapezoid_pulse',
                  funcs={'trapezoid_pulse.h': funcs})]
