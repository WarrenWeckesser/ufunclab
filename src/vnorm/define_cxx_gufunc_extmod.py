
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

RMS_DOCSTRING = """\
rms(z, /, ...)

Root-mean-square value of the 1-d array z.
"""

VDOT_DOCSTRING = """\
vdot(x, y, /, ...)

Vector dot product of the real floating point arrays x and y.
"""

vnorm_src_real = UFuncSource(
    funcname='vnorm_core_calc',
    typesignatures=['ff->f', 'dd->d', 'gg->g'],
)

vnorm_src_cplx = UFuncSource(
    funcname='cvnorm_core_calc',
    typesignatures=['Ff->f', 'Dd->d', 'Gg->g'],
)

vnorm = UFunc(
    name='vnorm',
    header='vnorm_gufunc.h',
    docstring=VNORM_DOCSTRING,
    signature='(n),()->()',
    sources=[vnorm_src_real, vnorm_src_cplx],
)

rms_src_real = UFuncSource(
    funcname='rms_core_calc',
    typesignatures=['f->f', 'd->d', 'g->g'],
)

rms_src_cplx = UFuncSource(
    funcname='crms_core_calc',
    typesignatures=['F->f', 'D->d', 'G->g'],
)

rms = UFunc(
    name='rms',
    header='vnorm_gufunc.h',
    docstring=RMS_DOCSTRING,
    signature='(n)->()',
    sources=[rms_src_real, rms_src_cplx],
)

vdot_src_real = UFuncSource(
    funcname='vdot_core_calc',
    typesignatures=['ff->f', 'dd->d', 'gg->g'],
)

vdot = UFunc(
    name='vdot',
    header='vnorm_gufunc.h',
    docstring=VDOT_DOCSTRING,
    signature='(n),(n)->()',
    sources=[vdot_src_real],
)

extmod = UFuncExtMod(
    module='_vnorm',
    docstring="This extension module defines the gufuncs 'vnorm' and 'vdot'.",
    ufuncs=[vnorm, rms, vdot],
)
