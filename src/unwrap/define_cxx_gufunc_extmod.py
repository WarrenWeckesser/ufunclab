
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


UNWRAP_DOCSTRING = """\
unwrap(x, period, /, ...)

Unwrap (aka lift) samples from a periodic domain to the real line.

This is like a pared down version of NumPy's `unwrap` function.  This
version works with floating point types only (single, double and long
double precision). Also, unlike `numpy.unwrap`, this function does not have
have the `discont` parameter.

Parameters
----------
x : array_like
    Input array
period : scalar
    Period of the periodic data.

Returns
-------
out : ndarray

Examples
--------
>>> import numpy as np
>>> from ufunclab import unwrap

>>> x = np.array([1.0, 10.0, 1.5, 2.0, -5.0])
>>> period = 4.0
>>> unwrap(x, period)
array([1. , 2. , 1.5, 2. , 3. ])
"""

unwrap_core_source = UFuncSource(
    funcname='unwrap_core',
    typesignatures=['ff->f', 'dd->d', 'gg->g'],
)

unwrap_gufunc = UFunc(
    name='unwrap',
    docstring=UNWRAP_DOCSTRING,
    header='unwrap_gufunc.h',
    signature='(n),() -> (n)',
    sources=[unwrap_core_source],
)


MODULE_DOCSTRING = """\
This module defines the unwrap function.
"""

extmod = UFuncExtMod(
    module='_unwrap',
    docstring=MODULE_DOCSTRING,
    ufuncs=[unwrap_gufunc],
)
