
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


KBNSUM_DOCSTRING = """\
kbnsum(x, /, ...)

Compute the Kahan-Babuska-Neumaier summation of the input `x`.

Parameters
----------
x : array_like
    Input array

Returns
-------
sum : ndarray
    Sum of the input.

Examples
--------
>>> import numpy as np
>>> from ufunclab import kbnsum

>>> x = np.array([100.0, -1e-13, -25.0, -4.5e-15, 5.0, -80, 1e-23])

The correct double precision floating point sum of `x` is -1.0449999999e-13.
The numpy `sum` function (using numpy 2.4.3) loses precision:

>>> np.sum(x)
np.float64(-9.947598299641402e-14)

For this example, `kbnsum()` gives the expected result:

>>> kbnsum(x)
np.float64(-1.0449999999e-13)
"""

kbnsum_src_real = UFuncSource(
    funcname='kbnsum_core_calc',
    typesignatures=['f->f', 'd->d', 'g->g'],
)

kbnsum = UFunc(
    name='kbnsum',
    header='kbnsum_gufunc.h',
    docstring=KBNSUM_DOCSTRING,
    signature='(n)->()',
    sources=[kbnsum_src_real],
)

extmod = UFuncExtMod(
    module='_kbnsum',
    docstring="This extension module defines the gufunc 'kbnsum'.",
    ufuncs=[kbnsum],
)
