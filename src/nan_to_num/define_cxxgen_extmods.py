
from ufunc_config_types import ExtMod, Func

nan_to_num_docstring = """\
nan_to_num(x, replacement, /, ...)

Replace occurrences of `nan` with `replacement`.

Parameters
----------
x : array_like
    Input values. Implemented for np.float32, np.float64 and np.longdouble.
replacement : array_like
    Value to replace nan.

Returns
-------
out : ndarray

Notes
-----

This is not the same as `np.nan_to_num`.  Unlike `np.nan_to_num`, this
function casts integer inputs to floating point, and does not handle
complex inputs. To operate in-place, use the `out` parameter (see the
example below).

Examples
--------
>>> import numpy as np
>>> from ufunclab import nan_to_num

>>> x = np.array([3.0, np.nan, 100.0, 0.25, np.nan])
>>> nan_to_num(x, -1.0)
array([  3.  ,  -1.  , 100.  ,   0.25,  -1.  ])

This operation is not in-place:

>>> x
array([  3.  ,    nan, 100.  ,   0.25,    nan])

To operate in-place, use the `out` parameter:

>>> nan_to_num(x, -1.0, out=x)
array([  3.  ,  -1.  , 100.  ,   0.25,  -1.  ])

>>> x
array([  3.  ,  -1.  , 100.  ,   0.25,  -1.  ])

"""

funcs = [
    Func(cxxname='nan_to_num',
         ufuncname='nan_to_num',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=nan_to_num_docstring),
]

extmods = [ExtMod(modulename='_nan_to_num',
                  funcs={'nan_to_num.h': funcs})]
