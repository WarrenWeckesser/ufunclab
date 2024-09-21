from itertools import product
import numpy as np
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


BINCOUNT_DOCSTRING = """\
bincount(x, /, ...)

Count nonnegative integers in `x`, a 1-d array of length n.
The `out` array *must* be given.
"""

BINCOUNTW_DOCSTRING = """\
bincountw(x, w, /, ...)

Count nonnegative integers in `x`, a 1-d array of length n.
If `x[k] = i`, `out[i]` is incremented by `w[k]`.
When `w` is all ones, `bincountw(x, w)` is the same as `bincount(x)`.
The `out` array *must* be given.
"""

int_types = [np.dtype('int8'), np.dtype('uint8'),
             np.dtype('int16'), np.dtype('uint16'),
             np.dtype('int32'), np.dtype('uint32'),
             np.dtype('int64'), np.dtype('uint64')]
float_types = [np.dtype('f'), np.dtype('d')]
input_types = product(int_types, int_types + float_types)
bincount_types = [f'{t.char}->p' for t in int_types]
bincountw_types = [f'{in1.char}{in2.char}->{in2.char}' for (in1, in2) in input_types]

bincount_src = UFuncSource(
    funcname='bincount_core_calc',
    typesignatures=bincount_types,
)

bincount = UFunc(
    name='bincount',
    header='bincount_gufunc.h',
    docstring=BINCOUNT_DOCSTRING,
    signature='(n)->(m)',
    sources=[bincount_src],
)

bincountw_src = UFuncSource(
    funcname='bincountw_core_calc',
    typesignatures=bincountw_types,
)

bincountw = UFunc(
    name='bincountw',
    header='bincount_gufunc.h',
    docstring=BINCOUNTW_DOCSTRING,
    signature='(n),(n)->(m)',
    sources=[bincountw_src],
)

extmod = UFuncExtMod(
    module='_bincount',
    docstring=("This extension module defines the gufuncs 'bincount' "
               "and 'bincountw'."),
    ufuncs=[bincount, bincountw],
)
