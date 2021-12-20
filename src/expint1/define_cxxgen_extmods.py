
from ufunc_config_types import ExtMod, Func


expint1_docstring = """\
expint1(x, /, ...)

Compute the exponential integral E1 of the input x.

Parameters
----------
x : array_like
    Input values

Returns
-------
out : ndarray
    The computed values of E1.

Examples
--------
>>> import numpy as np
>>> from ufunclab import expint1
>>> expint1([0.25, 2.5, 25])
array([1.04428263e+00, 2.49149179e-02, 5.34889976e-13])
"""

logexpint1_docstring = """\
logexpint1(x, /, ...)

Compute the log of the exponential integral E1 of the input x.

Parameters
----------
x : array_like
    Input values

Returns
-------
out : ndarray
    The computed values of log E1.

Examples
--------
>>> import numpy as np
>>> from ufunclab import logexpint1
>>> logexpint1([0.25, 2.5, 25])
array([1.04428263e+00, 2.49149179e-02, 5.34889976e-13])

``expint1(x)`` underflows to 0 for sufficiently large x:

>>> from ufunclab import expint1
>>> expint1([650, 700, 750, 800])
array([7.85247922e-286, 1.40651877e-307, 0.00000000e+000, 0.00000000e+000])

``logexpint1`` avoids the underflow by computing the logarithm of
the value:

>>> logexpint1([650, 700, 750, 800])
array([-656.47850729, -706.55250586, -756.62140388, -806.68585939])
"""


funcs = [
    Func(cxxname='expint1',
         ufuncname='expint1',
         types=['f->f', 'd->d', 'g->g'],
         docstring=expint1_docstring),
    Func(cxxname='logexpint1',
         ufuncname='logexpint1',
         types=['f->f', 'd->d', 'g->g'],
         docstring=logexpint1_docstring),
]

extmods = [ExtMod(modulename='_expint1',
                  funcs={'expint1.h': funcs})]
