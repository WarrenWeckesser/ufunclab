
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


SOFTMAX_DOCSTRING = """\
softmax(x, /, ...)

Compute the softmax function of the 1-d array `x`.
See https://en.wikipedia.org/wiki/Softmax_function for more information.

If `x` contains `nan` or more than one positive infinity, the result is
all `nan`.

Parameters
----------
x : array_like
    Input array

Returns
-------
out : ndarray

Examples
--------
>>> import numpy as np
>>> from ufunclab import softmax

>>> x = np.array([1.0, 10.0, 1.5, 2.0, -5.0])
>>> softmax(x)
array([1.23328081e-04, 9.99337792e-01, 2.03333631e-04, 3.35240482e-04,
       3.05699750e-07])
"""

softmax_core_source = UFuncSource(
    funcname='softmax_core',
    typesignatures=['f->f', 'd->d', 'g->g'],
)

softmax_gufunc = UFunc(
    name='softmax',
    docstring=SOFTMAX_DOCSTRING,
    header='softmax_gufunc.h',
    signature='(n) -> (n)',
    sources=[softmax_core_source],
)


MODULE_DOCSTRING = """\
This module defines the softmax function.
"""

extmod = UFuncExtMod(
    module='_softmax',
    docstring=MODULE_DOCSTRING,
    ufuncs=[softmax_gufunc],
)
