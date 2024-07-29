
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


CONVERT_TO_BASE_DOCSTRING = """\
convert_to_base(k, base, /, ...)

Convert the integer k to the given base.
"""

convert_to_base_src = UFuncSource(
    funcname='convert_to_base_core_calc',
    typesignatures=[
        'bb->b',
        'BB->B',
        'hh->h',
        'HH->H',
        'ii->i',
        'II->I',
        'll->l',
        'LL->L',
    ]
)


convert_to_base = UFunc(
    name='convert_to_base',
    header='convert_to_base_gufunc.h',
    docstring=CONVERT_TO_BASE_DOCSTRING,
    signature='(),()->(n)',
    sources=[convert_to_base_src],
)

extmod = UFuncExtMod(
    module='_convert_to_base',
    docstring=("This extension module defines the gufunc 'convert_to_base'."),
    ufuncs=[convert_to_base],
)
