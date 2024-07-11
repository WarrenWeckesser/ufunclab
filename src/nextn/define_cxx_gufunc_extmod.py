
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


NEXTN_GREATER_DOCSTRING = """\
nextn_greater(x, /, ...)

Return the next n floating point values greater than x.

The `out` parameter of this ufunc must be given.  The length
of `out` determines n.
"""

NEXTN_LESS_DOCSTRING = """\
nextn_less(x, /, ...)

Return the next n floating point values less than x.

The `out` parameter of this ufunc must be given.  The length
of `out` determines n.
"""



nextn_greater_src = UFuncSource(
    funcname='nextn_greater_core_calc',
    typesignatures=[
        'f->f',
        'd->d',
        'g->g',
    ]
)

nextn_greater = UFunc(
    name='nextn_greater',
    header='nextn_gufunc.h',
    docstring=NEXTN_GREATER_DOCSTRING,
    signature='()->(n)',
    sources=[nextn_greater_src],
)

nextn_less_src = UFuncSource(
    funcname='nextn_less_core_calc',
    typesignatures=[
        'f->f',
        'd->d',
        'g->g',
    ]
)

nextn_less = UFunc(
    name='nextn_less',
    header='nextn_gufunc.h',
    docstring=NEXTN_LESS_DOCSTRING,
    signature='()->(n)',
    sources=[nextn_less_src],
)


extmod = UFuncExtMod(
    module='_nextn',
    docstring=("This extension module defines the gufuncs 'nextn_greater' and "
               "'nextn_less'."),
    ufuncs=[nextn_greater, nextn_less],
)
