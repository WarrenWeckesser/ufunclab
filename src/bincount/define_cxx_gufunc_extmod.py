
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


BINCOUNT_DOCSTRING = """\
bincount(x, /, ...)

Count integer in `x`, a 1-d array of length n.
The `out` array *must* be given, and it must be a 1-d.
"""

bincount_src = UFuncSource(
    funcname='bincount_core_calc',
    typesignatures=[
        'b->P',
        'B->P',
        'h->P',
        'H->P',
        'i->P',
        'I->P',
        'l->P',
        'L->P',
        'q->P',
        'Q->P',
        'p->P',
        'P->P',
    ]
)

bincount = UFunc(
    name='bincount',
    header='bincount_gufunc.h',
    docstring=BINCOUNT_DOCSTRING,
    signature='(n)->(m)',
    sources=[bincount_src],
)

extmod = UFuncExtMod(
    module='_bincount',
    docstring=("This extension module defines the gufunc 'bincount'."),
    ufuncs=[bincount],
)
