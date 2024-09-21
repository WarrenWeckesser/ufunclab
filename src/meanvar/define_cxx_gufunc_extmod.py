import numpy as np
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


MEANVAR_DOCSTRING = """\
meanvar(x, ddof, /, ...)

Simultaneous mean and variance calculation for x.

Parameters
----------
x : array_like
    The array for which the statistics are computed.
ddof : int
    The 'delta' degress of freedom; see `numpy.var`.
    `ddof=1` gives the unbiased estimate of the variance.

Returns
-------
out : ndarray
    By default, the last dimension of `out` has length 2;
    the first value is the mean and the second value is the
    variance.

Examples
--------
>>> import numpy as np
>>> from ufunclab import meanvar
>>> meanvar([1, 2, 4, 5], 0)
array([3. , 2.5])

By default, `meanvar` acts on the last dimension of
multidimensional arrays:

>>> x = np.array([[1, 4, 4, 2, 1, 1, 2, 7],
...               [0, 0, 9, 4, 1, 0, 0, 1],
...               [8, 3, 3, 3, 3, 3, 3, 3],
...               [5, 5, 5, 5, 5, 5, 5, 5]])
>>> meanvar(x, 1)
array([[ 2.75 ,  4.5  ],
       [ 1.875, 10.125],
       [ 3.625,  3.125],
       [ 5.   ,  0.   ]])

The `axes` parameter can be used to change the axes on
which the function operates.  For example, to compute the
mean and variance along the first axis, but leave the result
(i.e. the mean and variance pairs) in the last dimension
of the output, use `axes=[0, (), 1]`:

>>> meanvar(x, 1, axes=[0, (), 1])
array([[ 3.5       , 13.66666667],
       [ 3.        ,  4.66666667],
       [ 5.25      ,  6.91666667],
       [ 3.5       ,  1.66666667],
       [ 2.5       ,  3.66666667],
       [ 2.25      ,  4.91666667],
       [ 2.5       ,  4.33333333],
       [ 4.        ,  6.66666667]])
"""

int_types = [np.dtype('int8'), np.dtype('uint8'),
             np.dtype('int16'), np.dtype('uint16'),
             np.dtype('int32'), np.dtype('uint32'),
             np.dtype('int64'), np.dtype('uint64')]
float_types = [np.dtype('f'), np.dtype('d')]
if np.dtype('d') != np.dtype('g'):
    float_types.append(np.dtype('g'))

int_type_sigs = [f'{t.char}p->d' for t in int_types]
float_type_sigs = [f'{t.char}p->{t.char}' for t in float_types]

ufunc_src = UFuncSource(
    funcname='meanvar_core',
    typesignatures=int_type_sigs + float_type_sigs,
)

ufunc = UFunc(
    name='meanvar',
    header='meanvar_gufunc.h',
    docstring=MEANVAR_DOCSTRING,
    signature='(n),()->(2)',
    sources=[ufunc_src],
    nonzero_coredims=['n'],  # n must be at least 1.
)

extmod = UFuncExtMod(
    module='_meanvar',
    docstring="This extension module defines the gufunc 'meanvar'.",
    ufuncs=[ufunc],
)
