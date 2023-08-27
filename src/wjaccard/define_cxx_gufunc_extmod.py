from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


WJACCARD_DOCSTRING = """\
wjaccard(x, y, /, *)

Weighted Jaccard index Jw(x, y):

    Jw(x, y) = min(x, y).sum() / max(x, y).sum()

Parameters
---------
x, y : array_like (1-d)
    Arrays for which the weighted Jaccard index is computed.
    The lengths of `x` and `y` must be the same, and must be
    at least 1.  An error is raised if either length is 0.

Returns
-------
Jw : scalar
    The weighted Jaccard index.
    `nan` is returned if either input contains `nan`.
    `nan` is also returned if `x` or `y` contains `inf` or `-inf`
    positioned such that the computed value is `inf/inf`.

Example
-------
>>> import numpy as np
>>> from ufunclab import wjaccard

>>> x = np.array([0.9, 1.0, 0.7, 0.0, 0.8, 0.6])
>>> y = np.array([0.3, 1.0, 0.9, 0.6, 1.0, 0.2])
>>> wjaccard(x, y)
0.6

>>> a = np.array([2, 3, 1, 22, 12, 10, 19, 4], dtype=np.uint8)
>>> b = np.array([5, 5, 8, 10, 10, 10, 2, 11], dtype=np.uint8)
>>> wjaccard(a, b)
>>> 0.45652173913043476
"""

wjaccard_integer_src = UFuncSource(
    funcname='wjaccard_integer_core',
    typesignatures=['bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d',
                    'll->d', 'LL->d'],
)

wjaccard_realtype_src = UFuncSource(
    funcname='wjaccard_realtype_core',
    typesignatures=['ff->f', 'dd->d'],
)

wjaccard_ufunc = UFunc(
    name='wjaccard',
    header='wjaccard_gufunc.h',
    docstring=WJACCARD_DOCSTRING,
    signature='(n),(n)->()',
    sources=[wjaccard_integer_src, wjaccard_realtype_src],
    nonzero_coredims=['n'],
)

extmod = UFuncExtMod(
    module='_wjaccard',
    docstring="This extension module defines the gufunc 'wjaccard'.",
    ufuncs=[wjaccard_ufunc],
)
