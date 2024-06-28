
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

TRI_AREA_INDEXED_DOCSTRING = """\
tri_area_indexed(p, i, /, ...)

Compute the area of the triangle formed by three points
in n-dimensional space.  `p` is an array of `m` points
in n-dimensional space, and `i` is a 1-d array of
three integers.  These integers are indices into `p`.

Parameters
----------
p : array_like, shape (m, n)
    Input array of points.
i : array_like, shape (3,)
    Indices into `i`.

Returns
-------
out : ndarray
    Area of the triangle formed by `p[i[0]]`, `p[i[1]]` and `p[i[2]]`.

Examples
--------

>>> import numpy as np
>>> from ufunclab import tri_area_indexed, tri_area

>>> p = np.array([[0.0, 0.0, 0.0, 6.0],
                  [1.0, 2.0, 3.0, 6.0],
                  [0.0, 2.0, 2.0, 6.0],
                  [1.5, 1.0, 2.5, 2.0],
                  [4.0, 1.0, 0.0, 2.5],
                  [2.0, 1.0, 2.0, 2.5]])
>>> tri_area_indexed(p, [0, 2, 3])
6.224949798994367
>>> tri_area(p[[0, 2, 3]])
6.224949798994367

Compute the areas of several triangles formed from points in `p`.
Note that the last two are the same triangles.

>>> tri_area_indexed(p, [[0, 2, 3], [1, 3, 4], [3, 4, 5], [-1, -2, -3]])
array([6.2249498 , 7.46449931, 0.70710678, 0.70710678])

"""

ta_ufunc_src = UFuncSource(
    funcname='tri_area_core_calc',
    typesignatures=['f->f', 'd->d', 'g->g'],
)

ta_ufunc = UFunc(
    name='tri_area',
    header='tri_area_gufunc.h',
    docstring=TRI_AREA_DOCSTRING,
    signature='(3,n)->()',
    sources=[ta_ufunc_src],
)

tai_ufunc_src = UFuncSource(
    funcname='tri_area_indexed_core_calc',
    typesignatures=['fp->f', 'dp->d', 'gp->g'],
)

tai_ufunc = UFunc(
    name='tri_area_indexed',
    header='tri_area_gufunc.h',
    docstring=TRI_AREA_INDEXED_DOCSTRING,
    signature='(m,n),(3)->()',
    sources=[tai_ufunc_src],
)

extmod = UFuncExtMod(
    module='_tri_area',
    docstring=("This extension module defines the gufuncs 'tri_area' "
               "and 'tri_area_indexed'."),
    ufuncs=[ta_ufunc, tai_ufunc],
)
