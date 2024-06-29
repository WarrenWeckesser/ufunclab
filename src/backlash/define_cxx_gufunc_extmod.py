

from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


BACKLASH_DOCSTRING = """\
backlash(x, deadband, initial, /, ...)

Compute the backlash signal of the input signal x.

Parameters
----------
x : array_like
    Input signal
deadband : scalar
    Width of the deadband of the backlash process.
initial : scalar
    Initial state of the output.

Returns
-------
out : ndarray
    Output of the backlash process.

Examples
--------
>>> x = np.array([0, 1, 1.1, 1.0, 1.5, 1.4, 1.2, 0.5])
>>> backlash(x, 0.4, 0.0)
array([0. , 0.8, 0.9, 0.9, 1.3, 1.3, 1.3, 0.7])
"""


backlash_core_source = UFuncSource(
    funcname='backlash_core',
    typesignatures=['fff->f', 'ddd->d', 'ggg->g'],
)

backlash_gufunc = UFunc(
    name='backlash',
    docstring=BACKLASH_DOCSTRING,
    header='backlash_gufunc.h',
    signature='(n),(),() -> (n)',
    sources=[backlash_core_source],
)


MODULE_DOCSTRING = """\
This module defines the backlash function.
"""

extmod = UFuncExtMod(
    module='_backlash',
    docstring=MODULE_DOCSTRING,
    ufuncs=[backlash_gufunc],
)
