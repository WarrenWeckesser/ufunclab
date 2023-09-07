
from ufunc_config_types import ExtMod, Func


deadzone_docstring = """\
deadzone(x, low, high, /, ...)

Compute the deadzone transform of the input signal x.
The function is also known as a soft threshold.

Parameters
----------
x : array_like
    Input signal
low : array_like
    Low end of the dead zone.
high : array_like
    High end of the dead zone.

Returns
-------
out : ndarray
    Output of the deadzone transform.

Notes
-----
The function expects ``low <= high``.  It does not check
that this condition is satisfied.

Examples
--------
>>> x = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5])
>>> deadzone(x, -0.25, 0.1)
array([-0.75, -0.5 , -0.25,  0.  ,  0.  ,  0.15,  0.4 ])
"""


funcs = [
    Func(cxxname='deadzone',
         ufuncname='deadzone',
         types=['fff->f', 'ddd->d'], # 'ggg->g'],
         docstring=deadzone_docstring),
]

extmods = [ExtMod(modulename='_deadzone',
                  funcs={'deadzone.h': funcs})]
