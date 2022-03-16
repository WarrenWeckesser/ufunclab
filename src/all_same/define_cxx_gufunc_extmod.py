
from ufunc_config_types import GUFuncExtMod

ALL_SAME_DOCSTRING = """\
all_same(x, /, ...)

Test for all values being the same.
"""


extmod = GUFuncExtMod(
    module='_all_same',
    ufuncname='all_same',
    docstring=ALL_SAME_DOCSTRING,
    signature='(n)->()',
    corefuncs={'all_same_core': ['b->?','B->?', 'h->?', 'H->?', 'i->?', 'I->?',
                                 'l->?', 'L->?', 'f->?', 'd->?', 'g->?'],
               'all_same_core_object': ['O->?'],
               },
    header='all_same_gufunc.h',
)
