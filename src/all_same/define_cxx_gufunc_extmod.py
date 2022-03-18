from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource

ALL_SAME_DOCSTRING = """\
all_same(x, /, ...)

Test for all values being the same.
"""

all_same_core = UFuncSource(
    funcname='all_same_core',
    typesignatures=['b->?','B->?', 'h->?', 'H->?', 'i->?', 'I->?',
                    'l->?', 'L->?', 'f->?', 'd->?', 'g->?'],
)

all_same_core_object = UFuncSource(
    funcname='all_same_core_object',
    typesignatures=['O->?'],
)

ufunc = UFunc(
    name="all_same",
    header='all_same_gufunc.h',
    docstring=ALL_SAME_DOCSTRING,
    signature='(n)->()',
    sources=[all_same_core, all_same_core_object],
)

extmod = UFuncExtMod(
    module='_all_same',
    docstring="This extension module defines the gufunc 'all_same'.",
    ufuncs=[ufunc],
)
