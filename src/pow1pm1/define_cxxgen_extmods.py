
from ufunc_config_types import ExtMod, Func


pow1pm1_docstring = """\
pow1pm1(x, y, /, ...)

Compute `(1 + x)**y - 1` for `x >= -1`.

The calculation is formulated to avoid loss of precision when
`(1 + x)**y` is close to 1.

`nan` is returned when `x < -1`.

The function follows the widely used convention that `pow(0, 0)`
is 1, so `pow1pm1(-1, 0)` is 0.0.

Parameters
----------
x : array_like
    Input values
y : array_like
    Input values

Returns
-------
out : ndarray
    The computed values of (1 + x)**y - 1.

Examples
--------
>>> import numpy as np
>>> from ufunclab import pow1pm1

>>> x = 3e-13
>>> y = 0.25
>>> pow1pm1(x, y)
7.499999999999156e-14

The naive expression is inaccurate:

>>> (1 + x)**y - 1
7.505107646466058e-14
"""

funcs = [
    Func(cxxname='pow1pm1',
         ufuncname='pow1pm1',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=pow1pm1_docstring),
]

extmods = [ExtMod(modulename='_pow1pm1',
                  funcs={'pow1pm1.h': funcs})]
