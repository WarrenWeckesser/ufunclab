import numpy as np
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


PERCENTILEOFSCORE_DOCSTRING = """\
percentileofscore(x, score, kind, /, ...)

See `scipy.stats.percentileofscore`.
"""

input_types = [np.dtype('int8'), np.dtype('uint8'),
               np.dtype('int16'), np.dtype('uint16'),
               np.dtype('int32'), np.dtype('uint32'),
               np.dtype('int64'), np.dtype('uint64'),
               np.dtype('f'), np.dtype('d')]

type_chars = [typ.char for typ in input_types]
type_sigs = [f'{c}{c}p->d' for c in type_chars]

percentileofscore_src = UFuncSource(
    funcname='percentileofscore_core_calc',
    typesignatures=type_sigs,
)

percentileofscore = UFunc(
    name='percentileofscore',
    header='percentileofscore_gufunc.h',
    docstring=PERCENTILEOFSCORE_DOCSTRING,
    signature='(n),(),()->()',
    sources=[percentileofscore_src],
)


extmod = UFuncExtMod(
    module='_percentileofscore',
    docstring=("This extension module defines the gufuncs 'percentileofscore'."),
    ufuncs=[percentileofscore],
    # The call `status = add_percentileofscore_constants(module);` will be
    # added to the end of the module init function.
    extra_module_funcs=['add_percentileofscore_kind_constants']
)
