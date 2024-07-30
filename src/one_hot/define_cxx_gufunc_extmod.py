
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


ONE_HOT_DOCSTRING = """\
one_hot(k, /, ...)

Fill the 1-d output with zeros except at k, where the output is 1.

The `out` parameter of this ufunc must be given.  The length
of `out` determines n.
"""

one_hot_src = UFuncSource(
    funcname='one_hot_core_calc',
    typesignatures=[
        'p->p',
    ]
)

one_hot = UFunc(
    name='one_hot',
    header='one_hot_gufunc.h',
    docstring=ONE_HOT_DOCSTRING,
    signature='()->(n)',
    sources=[one_hot_src],
)


extmod = UFuncExtMod(
    module='_one_hot',
    docstring=("This extension module defines the gufunc 'one_hot'."),
    ufuncs=[one_hot],
)
