
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


TRI_AREA_DOCSTRING = """\
tri_area(p, /, ...)

Compute the area of the triangle formed by three points
in n-dimensional space.

Parameters
----------
x : array_like, shape (..., 3, n)
    Input array

Returns
-------
out : ndarray
    Area(s) of the triangle(s).

Examples
--------
>>> import numpy as np
>>> from ufunclab import tri_area

`p` has shape (2, 3, 4). It contains the vertices
of two triangles in 4-dimensional space.

>>> p = np.array([[[0.0, 0.0, 0.0, 6.0],
                   [1.0, 2.0, 3.0, 6.0],
                   [0.0, 2.0, 2.0, 6.0]],
                  [[1.5, 1.0, 2.5, 2.0],
                   [4.0, 1.0, 0.0, 2.5],
                   [2.0, 1.0, 2.0, 2.5]]])
>>> tri_area(p)
array([1.73205081, 0.70710678])
"""

ufunc_src = UFuncSource(
    funcname='tri_area_core_calc',
    typesignatures=['f->f', 'd->d', 'g->g'],
)


ufunc = UFunc(
    name='tri_area',
    header='tri_area_gufunc.h',
    docstring=TRI_AREA_DOCSTRING,
    signature='(3,n)->()',
    sources=[ufunc_src],
)

extmod = UFuncExtMod(
    module='_tri_area',
    docstring="This extension module defines the gufunc 'tri_area'.",
    ufuncs=[ufunc],
)
