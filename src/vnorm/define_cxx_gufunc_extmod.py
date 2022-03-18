
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


VNORM_DOCSTRING = """\
vnorm(x, p, /, ...)

Compute the vector norm of the input signal x.

Parameters
----------
x : array_like
    Input array
p : scalar or array_like
    Order of the norm to be computed (e.g. p=2 is the
    standard Euclidean norm). Must be greater than 0.

Returns
-------
out : ndarray
    Vector norm(s) of the input.

Examples
--------
>>> x = np.array([30.0, 40.0])
>>> vnorm(x, 2)
50.0
>>> z = np.array([[-2j, 3+4j, 0, 14], [3, -12j, 3j, 2-2j]])
>>> vnorm(z, 2, axis=1)
array([15.        , 13.03840481])
>>> vnorm(z, 2, axis=0)
array([ 3.60555128, 13.        ,  3.        , 14.28285686])
"""

ufunc_src_real = UFuncSource(
    funcname='vnorm_core_calc',
    typesignatures=['ff->f', 'dd->d', 'gg->g'],
)

ufunc_src_cplx = UFuncSource(
    funcname='cvnorm_core_calc',
    typesignatures=['Ff->f', 'Dd->d', 'Gg->g'],
)

ufunc = UFunc(
    name='vnorm',
    header='vnorm_gufunc.h',
    docstring=VNORM_DOCSTRING,
    signature='(n),()->()',
    sources=[ufunc_src_real, ufunc_src_cplx],
)

extmod = UFuncExtMod(
    module='_vnorm',
    docstring="This extension module defines the gufunc 'vnorm'.",
    ufuncs=[ufunc],
)