from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource

ARGMIN_DOCSTRING = """\
argmin(x, /, ...)

Compute the index of the minimum of the input.
"""

ARGMAX_DOCSTRING = """\
argmax(x, /, ...)

Compute the index of the maximum of the input.
"""


MINMAX_DOCSTRING = """\
minmax(x, /, ...)

Compute minimum and maximum of the input.
"""

ARGMINMAX_DOCSTRING = """\
argminmax(x, /, ...)

Compute indices of the minimum and maximum of the input.
"""

MIN_ARGMIN_DOCSTRING = """\
min_argmin(x, /, ...)

Compute the minimum and index of the minimum of the input.
"""

MAX_ARGMAX_DOCSTRING = """\
max_argmax(x, /, ...)

Compute the maximum and index of the maximum of the input.
"""


argmin_core = UFuncSource(
    funcname='argmin_core',
    typesignatures=['b->p', 'B->p',
                    'h->p', 'H->p',
                    'i->p', 'I->p',
                    'l->p', 'L->p',
                    'q->p', 'Q->p',
                    'f->p', 'd->p', 'g->p'],
)

argmin_object_core = UFuncSource(
    funcname='argmin_object_core',
    typesignatures=['O->p'],
)

argmax_core = UFuncSource(
    funcname='argmax_core',
    typesignatures=['b->p', 'B->p',
                    'h->p', 'H->p',
                    'i->p', 'I->p',
                    'l->p', 'L->p',
                    'q->p', 'Q->p',
                    'f->p', 'd->p', 'g->p'],
)

argmax_object_core = UFuncSource(
    funcname='argmax_object_core',
    typesignatures=['O->p'],
)

minmax_core = UFuncSource(
    funcname='minmax_core',
    typesignatures=['b->b', 'B->B',
                    'h->h', 'H->H',
                    'i->i', 'I->I',
                    'l->l', 'L->L',
                    'q->q', 'Q->Q',
                    'f->f', 'd->d', 'g->g'],
)

minmax_object_core = UFuncSource(
    funcname='minmax_object_core',
    typesignatures=['O->O'],
)

argminmax_core = UFuncSource(
    funcname='argminmax_core',
    typesignatures=['b->p', 'B->p',
                    'h->p', 'H->p',
                    'i->p', 'I->p',
                    'l->p', 'L->p',
                    'q->p', 'Q->p',
                    'f->p', 'd->p', 'g->p'],
)

argminmax_object_core = UFuncSource(
    funcname='argminmax_object_core',
    typesignatures=['O->p'],
)

min_argmin_core = UFuncSource(
    funcname='min_argmin_core',
    typesignatures=['b->bp', 'B->Bp',
                    'h->hp', 'H->Hp',
                    'i->ip', 'I->Ip',
                    'l->lp', 'L->Lp',
                    'q->qp', 'Q->Qp',
                    'f->fp', 'd->dp', 'g->gp'],
)

min_argmin_object_core = UFuncSource(
    funcname='min_argmin_object_core',
    typesignatures=['O->Op'],
)

max_argmax_core = UFuncSource(
    funcname='max_argmax_core',
    typesignatures=['b->bp', 'B->Bp',
                    'h->hp', 'H->Hp',
                    'i->ip', 'I->Ip',
                    'l->lp', 'L->Lp',
                    'q->qp', 'Q->Qp',
                    'f->fp', 'd->dp', 'g->gp'],
)

max_argmax_object_core = UFuncSource(
    funcname='max_argmax_object_core',
    typesignatures=['O->Op'],
)

argmin_ufunc = UFunc(
    name="argmin",
    header="minmax_gufunc.h",
    docstring=ARGMIN_DOCSTRING,
    signature='(n)->()',
    sources=[argmin_core, argmin_object_core],
    nonzero_coredims=['n'],
)

argmax_ufunc = UFunc(
    name="argmax",
    header="minmax_gufunc.h",
    docstring=ARGMAX_DOCSTRING,
    signature='(n)->()',
    sources=[argmax_core, argmax_object_core],
    nonzero_coredims=['n'],
)


minmax_ufunc = UFunc(
    name="minmax",
    header='minmax_gufunc.h',
    docstring=MINMAX_DOCSTRING,
    signature='(n)->(2)',
    sources=[minmax_core, minmax_object_core],
    nonzero_coredims=['n'],
)

argminmax_ufunc = UFunc(
    name="argminmax",
    header='minmax_gufunc.h',
    docstring=ARGMINMAX_DOCSTRING,
    signature='(n)->(2)',
    sources=[argminmax_core, argminmax_object_core],
    nonzero_coredims=['n'],
)

min_argmin_ufunc = UFunc(
    name="min_argmin",
    header="minmax_gufunc.h",
    docstring=MIN_ARGMIN_DOCSTRING,
    signature='(n)->(),()',
    sources=[min_argmin_core, min_argmin_object_core],
    nonzero_coredims=['n'],
)

max_argmax_ufunc = UFunc(
    name="max_argmax",
    header="minmax_gufunc.h",
    docstring=MAX_ARGMAX_DOCSTRING,
    signature='(n)->(),()',
    sources=[max_argmax_core, max_argmax_object_core],
    nonzero_coredims=['n'],
)

extmod = UFuncExtMod(
    module='_minmax',
    docstring="This extension module defines the various min-max gufuncs.",
    ufuncs=[argmin_ufunc, argmax_ufunc, minmax_ufunc, argminmax_ufunc,
            min_argmin_ufunc, max_argmax_ufunc],
)
