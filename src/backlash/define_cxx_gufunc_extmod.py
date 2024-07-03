

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

BACKLASH_SUM_DOCSTRING = """\
backlash_sum(x, w, deadband, initial, /, ...)

Compute the linear combination of several backlash processes. This
operation is also known as the Prandtl-Ishlinskii hysteresis model.

Parameters
----------
x : array_like, length n
    The input signal
w : array_like, length m
    The weights of the backlash operators.
deadband : array_like, length m
    The deadband values of the backlash operators.
initial : array_like, length m
    The initial values of the backlash operators.

Returns
-------
out : ndarray, same shape as `x`
    Output of the Prandtl-Ishlinskii hysteresis model.
final : ndarray, same shape as `initial`
    State of the individual backlash processes.

Examples
--------
>>> import numpy as np
>>> from ufunclab import backlash_sum

>>> x = np.array([0.0, 0.2, 0.5, 1.1, 1.25, 1.0, 0.2, -1])

Here the weights are all the same and happen to sum to 1, but that
is not required in general.

>>> w = np.array([0.25, 0.25, 0.25, 0.25])
>>> deadband = np.array([0.2, 0.4, 0.6, 0.8])
>>> initial = np.zeros(4)

>>> y, final = backlash_sum(x, w, deadband, initial)
>>> y
array([ 0.    ,  0.025 ,  0.25  ,  0.85  ,  1.    ,  0.9875,  0.45  , -0.75  ])

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


backlash_sum_core_source = UFuncSource(
    funcname='backlash_sum_core',
    typesignatures=['ffff->ff', 'dddd->dd', 'gggg->gg'],
)

backlash_sum_gufunc = UFunc(
    name='backlash_sum',
    docstring=BACKLASH_SUM_DOCSTRING,
    header='backlash_gufunc.h',
    signature='(n),(m),(m),(m) -> (n),(m)',
    sources=[backlash_sum_core_source],
)

MODULE_DOCSTRING = """\
This module defines the backlash and backlash_sum functions.
"""

extmod = UFuncExtMod(
    module='_backlash',
    docstring=MODULE_DOCSTRING,
    ufuncs=[backlash_gufunc, backlash_sum_gufunc],
)
