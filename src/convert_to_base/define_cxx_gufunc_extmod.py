import numpy as np
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


CONVERT_TO_BASE_DOCSTRING = """\
convert_to_base(k, base, /, ...)

Convert the integer k to the given base.
"""

int_types = [np.dtype('int8'), np.dtype('uint8'),
             np.dtype('int16'), np.dtype('uint16'),
             np.dtype('int32'), np.dtype('uint32'),
             np.dtype('int64'), np.dtype('uint64')]
type_sigs = [f'{t.char}{t.char}->{t.char}' for t in int_types]

convert_to_base_src = UFuncSource(
    funcname='convert_to_base_core_calc',
    typesignatures=type_sigs,
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
