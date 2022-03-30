
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


MAD_DOCSTRING = """\
mad(x, unbiased, /, ...)

Mean absolute difference.
"""

RMAD_DOCSTRING = """\
rmad(x, unbiased, /, ...)

Relative mean absolute difference.
"""


GINI_DOCSTRING = """\
gini(x, unbiased, /, ...)

Gini coefficient.
"""


mad_core_ufunc_src = UFuncSource(
    funcname='mad_core',
    typesignatures=['f?->f', 'd?->d', 'g?->g'],
)

mad_ufunc = UFunc(
    name='mad',
    header='mad_gufunc.h',
    docstring=MAD_DOCSTRING,
    signature='(n),()->()',
    sources=[mad_core_ufunc_src],
)


rmad_core_ufunc_src = UFuncSource(
    funcname='rmad_core',
    typesignatures=['f?->f', 'd?->d', 'g?->g'],
)

rmad_ufunc = UFunc(
    name='rmad',
    header='mad_gufunc.h',
    docstring=RMAD_DOCSTRING,
    signature='(n),()->()',
    sources=[rmad_core_ufunc_src],
)


gini_core_ufunc_src = UFuncSource(
    funcname='gini_core',
    typesignatures=['f?->f', 'd?->d', 'g?->g'],
)

gini_ufunc = UFunc(
    name='gini',
    header='mad_gufunc.h',
    docstring=GINI_DOCSTRING,
    signature='(n),()->()',
    sources=[gini_core_ufunc_src],
)


extmod = UFuncExtMod(
    module='_mad',
    docstring=("This extension module defines the gufuncs\n"
               "mad, rmad, and gini."),
    ufuncs=[mad_ufunc, rmad_ufunc, gini_ufunc],
)
