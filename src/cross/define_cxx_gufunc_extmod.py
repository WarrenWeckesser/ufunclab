
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


CROSS3_DOCSTRING = """\
cross3(u, v, /, ...)

Compute the 3-d vector cross product of u and v.

Parameters
----------
u : array_like, shape (..., 3)
    Input array
v : array_like, shape (..., 3)
    Input array

Returns
-------
out : ndarray, shape (..., 3)
    Cross product of u and v.

See Also
--------
ufunclab.cross2
numpy.cross

Examples
--------
>>> from ufunclab import cross3
>>> cross3([1, 2, -2], [5, 3, 1])
array([  8, -11,  -7])
>>> cross3([[1, 2, -2], [6, 0, 2]], [[5, 3, 1], [2, 2, 3]])
array([[  8, -11,  -7],
       [ -4, -14,  12]])

"""

CROSS2_DOCSTRING = """\
cross2(u, v, /, ...)

Compute the cross product of 2-d vectors u and v.
The result is a scalar.

Parameters
----------
u : array_like, shape (..., 2)
    Input array
v : array_like, shape (..., 2)
    Input array

Returns
-------
out : scalar or ndarray, shape (...)
    Cross product of u and v.

See Also
--------
ufunclab.cross3
numpy.cross

Examples
--------
>>> from ufunclab import cross2
>>> cross2([1, 2], [5, 3])
-7
>>> cross2([[1, 2], [6, 0]], [[5, 3], [2, 3]])
array([-7, 18])
"""

cross3_src = UFuncSource(
    funcname='cross3_core_calc',
    typesignatures=['ii->i', 'll->l', 'ff->f', 'dd->d', 'gg->g'],
)

cross3_src_object = UFuncSource(
    funcname='cross3_core_calc_object',
    typesignatures=['OO->O'],
)

cross3 = UFunc(
    name='cross3',
    header='cross_gufunc.h',
    docstring=CROSS3_DOCSTRING,
    signature='(3),(3)->(3)',
    sources=[cross3_src, cross3_src_object],
)

cross2_src = UFuncSource(
    funcname='cross2_core_calc',
    typesignatures=['ii->i', 'll->l', 'ff->f', 'dd->d', 'gg->g'],
)

cross2_src_object = UFuncSource(
    funcname='cross2_core_calc_object',
    typesignatures=['OO->O'],
)

cross2 = UFunc(
    name='cross2',
    header='cross_gufunc.h',
    docstring=CROSS2_DOCSTRING,
    signature='(2),(2)->()',
    sources=[cross2_src, cross2_src_object],
)

extmod = UFuncExtMod(
    module='_cross',
    docstring="This extension module defines the gufuncs 'cross3' and 'cross2'.",
    ufuncs=[cross3, cross2],
)
