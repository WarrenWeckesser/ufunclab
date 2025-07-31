
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


FILLNAN1D_DOCSTRING = """\
fillnan1d(x, /, ...)

Replace `nan` in `x` by using linear interpolation.
`nan` values at either end of `x` are replaced by the
nearest non-`nan` value.
An array of all `nan` is returned as all `nan`.

Parameters
----------
x : array_like
    Input array

Returns
-------
out : ndarray
    Output array, with `nan` values replaced.

Examples
--------
>>> import numpy as np
>>> from ufunclab import fillnan1d
>>> x = np.array([1, np.nan, 1.5, 2.0, np.nan])
>>> fillnan1d(x)
array([1.  , 1.25, 1.5 , 2.  , 2.  ])
"""

fillnan1d_core_source = UFuncSource(
    funcname='fillnan1d_core',
    typesignatures=['f->f', 'd->d', 'g->g'],
)

fillnan1d_gufunc = UFunc(
    name='fillnan1d',
    docstring=FILLNAN1D_DOCSTRING,
    header='fillnan1d_gufunc.h',
    signature='(n) -> (n)',
    sources=[fillnan1d_core_source],
)


MODULE_DOCSTRING = """\
This module defines the fillnan1d function.
"""

extmod = UFuncExtMod(
    module='_fillnan1d',
    docstring=MODULE_DOCSTRING,
    ufuncs=[fillnan1d_gufunc],
)
